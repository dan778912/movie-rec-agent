# Prediction outputs

Generated prediction files are ignored by Git. Each method should write JSONL with one record per user containing `user_id`, `split`, `method`, and ranked `recommendations` with `movie_id`, `score`, and `rank`. See [Section 8 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).
