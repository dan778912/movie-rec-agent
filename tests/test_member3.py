import csv
import json
from pathlib import Path
import tempfile
import unittest

from src.baselines.audit import prepare, summarize
from src.baselines.common import jsonl, read_config
from src.baselines.checks import check_training, check_popularity
from src.baselines.popularity import predict
from src.baselines.training import build_examples
from src.data.build import sha256, write_jsonl


class Member3Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / 'data'
        self.data.mkdir()
        self.output = Path(self.temp.name) / 'out'
        self.output.mkdir()
        self.config = read_config(Path('configs/member3_v1.json'))
        self.config.update(minimum_history=1, training_user_limit=10, examples_per_user=20)
        docs = [{'movie_id': i, 'title': f'Movie {i}', 'genres': ['Drama'],
                 'genome_tags': ['warm'], 'user_tags': [], 'text': f'Movie {i}. Drama.'}
                for i in range(1, 8)]
        write_jsonl(self.data / 'movie_documents.jsonl', docs)
        with (self.data / 'train_ratings.csv').open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['user_id', 'movie_id', 'rating', 'timestamp'])
            for uid in (1, 2, 3):
                # Equal-time targets (2/3), ambiguous target (4), outside catalog (99).
                for mid, rating, ts in [(1, 5, 1), (2, 4, 2), (3, 2, 2), (4, 3.5, 3),
                                        (99, 5, 4), (5, 1, 5), (6, 4, 6)]:
                    writer.writerow([uid, mid, rating, ts])
        with (self.data / 'catalog.csv').open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['movie_id', 'train_positives'])
            writer.writerows([(1, 10), (2, 8), (3, 8), (4, 0)])
        for split, watched in [('validation', [1, 4]), ('test', [1, 2, 4])]:
            write_jsonl(self.data / f'{split}_profiles.jsonl', [
                {'user_id': 1, 'text': 'Likes drama', 'history_movie_ids': watched}])
        (self.data / 'statistics.json').write_text(json.dumps({'validation_cutoff_unix': 10}))
        self.refresh_manifest()

    def refresh_manifest(self):
        manifest = {'version': 'fixture', 'config': {'liked_movies_per_profile': 10,
                    'disliked_movies_per_profile': 5}, 'file_sha256': {}}
        for p in self.data.iterdir():
            if p.name != 'manifest.json':
                manifest['file_sha256'][p.name] = sha256(p)
        (self.data / 'manifest.json').write_text(json.dumps(manifest))

    def test_training_labels_and_strictly_earlier_history(self):
        out = self.output / 'training.jsonl'
        report = build_examples(self.data, out, self.config)
        rows = [r for r in jsonl(out) if r['user_id'] == 1]
        self.assertEqual([r['movie_id'] for r in rows], [2, 3, 5, 6])
        self.assertEqual(report['exported_labels'], {'relevant': 6, 'not_relevant': 6})
        self.assertEqual(report['training_population']['skipped_ambiguous'], 3)
        self.assertEqual(report['training_population']['skipped_outside_catalog'], 3)
        self.assertEqual(rows[0]['profile'], rows[1]['profile'])
        self.assertEqual(rows[0]['profile'], 'Highly rated movies: Movie 1 (5/5; genres: Drama; tags: warm).')
        self.assertEqual(rows[2]['history_count'], 5)  # Includes ambiguous/out-of-catalog history.
        self.assertEqual(rows[2]['negative_source'], 'explicit_rating')
        self.assertEqual(rows[3]['negative_source'], None)
        for row in rows:
            self.assertLess(row['history_latest_timestamp'], row['target_timestamp'])
            selected = row['liked_movies'] + row['disliked_movies']
            self.assertNotIn(row['movie_id'], [r['movie_id'] for r in selected])
            self.assertTrue(all(r['timestamp'] < row['target_timestamp'] for r in selected))
        before = out.read_bytes()
        build_examples(self.data, out, self.config)
        self.assertEqual(before, out.read_bytes())

    def test_sampling_is_bounded_and_reproducible(self):
        self.config.update(training_user_limit=2, examples_per_user=2)
        out = self.output / 'small.jsonl'
        report = build_examples(self.data, out, self.config)
        self.assertEqual(report['examples'], 4)
        self.assertEqual(report['selected_users'], 2)
        before = out.read_bytes()
        build_examples(self.data, out, self.config)
        self.assertEqual(before, out.read_bytes())

    def test_future_rows_are_rejected(self):
        path = self.data / 'train_ratings.csv'
        with path.open('a') as f:
            f.write('3,7,5,10\n')
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'non-training'):
            build_examples(self.data, self.output / 'bad.jsonl', self.config)

    def test_unsorted_training_history_is_rejected(self):
        path = self.data / 'train_ratings.csv'
        rows = path.read_text().splitlines()
        rows[2], rows[4] = rows[4], rows[2]
        path.write_text('\n'.join(rows) + '\n')
        self.refresh_manifest()
        with self.assertRaisesRegex(ValueError, 'Out-of-order'):
            build_examples(self.data, self.output / 'bad-order.jsonl', self.config)

    def test_input_integrity(self):
        with (self.data / 'catalog.csv').open('a') as f:
            f.write('5,100\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            predict(self.data, self.output / 'bad.jsonl', self.config)

    def test_b0_watched_items_ties_and_short_catalog(self):
        out = self.output / 'b0.jsonl'
        report = predict(self.data, out, self.config)
        row = next(jsonl(out))
        self.assertEqual(row['method'], 'popularity')
        self.assertEqual([r['movie_id'] for r in row['recommendations']], [2, 3])
        self.assertEqual([r['rank'] for r in row['recommendations']], [1, 2])
        self.assertEqual(report['short_lists'], 1)
        self.assertEqual(report['users'], 1)
        # Test-stage removal must include validation history as well.
        predict(self.data, out, self.config, split='test')
        self.assertEqual([r['movie_id'] for r in next(jsonl(out))['recommendations']], [3])

    def test_b0_fixed_subset_rejects_missing_users(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            predict(self.data, self.output / 'b0.jsonl', self.config, user_ids={99})

    def test_training_checker_catches_wrong_labels_and_future_profile(self):
        out = self.output / 'checked-training.jsonl'
        build_examples(self.data, out, self.config)
        report = check_training(self.data, out, self.config)
        self.assertEqual(report['examples_checked'], 12)
        rows = list(jsonl(out))
        rows[0]['label'] = 'not_relevant'
        write_jsonl(out, rows)
        with self.assertRaisesRegex(ValueError, 'Incorrect label'):
            check_training(self.data, out, self.config)
        rows[0]['label'] = 'relevant'
        rows[0]['history_latest_timestamp'] = rows[0]['target_timestamp']
        write_jsonl(out, rows)
        with self.assertRaisesRegex(ValueError, 'Future history'):
            check_training(self.data, out, self.config)

    def test_b0_checker_catches_watched_recommendations_and_missing_users(self):
        out = self.output / 'checked-b0.jsonl'
        predict(self.data, out, self.config)
        report = check_popularity(self.data, out, self.config)
        self.assertEqual(report['recommendations_checked'], 2)
        rows = list(jsonl(out))
        rows[0]['recommendations'][0]['movie_id'] = 1
        write_jsonl(out, rows)
        with self.assertRaisesRegex(ValueError, 'watched removal'):
            check_popularity(self.data, out, self.config)
        write_jsonl(out, [])
        with self.assertRaisesRegex(ValueError, 'Missing B0'):
            check_popularity(self.data, out, self.config)

    def test_audit_sampling_and_manual_summary(self):
        predictions = self.output / 'p1.jsonl'
        write_jsonl(predictions, [{'user_id': 1, 'split': 'validation', 'recommendations': [
            {'movie_id': i, 'rank': i, 'explanation': 'Drama suits your interests.'} for i in range(1, 8)]}])
        self.config['audit_sample_size'] = 3
        packet = self.output / 'audit.jsonl'
        report = prepare(self.data, predictions, packet, self.config)
        self.assertEqual(report['status'], 'pending_manual_review')
        self.assertEqual(report['sample_size'], 3)
        before = packet.read_bytes()
        prepare(self.data, predictions, packet, self.config)
        self.assertEqual(before, packet.read_bytes())
        summary = summarize(packet, self.output / 'summary.json')
        self.assertEqual(summary['reviewed'], 0)
        self.assertIsNone(summary['unsupported_fraction_of_reviewed'])
        rows = list(jsonl(packet))
        for row in rows:
            row['verdict'] = 'supported'
        rows[0]['verdict'] = 'unsupported'
        rows[0]['unsupported_claims'] = ['Invented actor credit']
        write_jsonl(packet, rows)
        summary = summarize(packet, self.output / 'summary.json')
        self.assertEqual(summary['status'], 'complete')
        self.assertAlmostEqual(summary['unsupported_fraction_of_reviewed'], 1 / 3)
        self.assertEqual(summary['verdict_counts']['supported'], 2)

    def test_audit_does_not_claim_missing_explanations_are_supported(self):
        pred = self.output / 'empty.jsonl'
        write_jsonl(pred, [{'user_id': 1, 'split': 'validation', 'recommendations': [
            {'movie_id': 2, 'rank': 1}]}])
        with self.assertRaisesRegex(ValueError, 'No explanations'):
            prepare(self.data, pred, self.output / 'empty-audit.jsonl', self.config)


if __name__ == '__main__':
    unittest.main()
