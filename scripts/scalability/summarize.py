"""Summarize saved Locust output; rejected requests are not successful throughput."""
import argparse
import csv
import json
from pathlib import Path


def summarize(root):
    output=[]
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory/'environment.json').exists():continue
        telemetry=[json.loads(line) for line in (directory/'pg-stat-activity.jsonl').read_text().splitlines()]
        probes={r['users']:r['recovery_probe'] for r in json.loads((directory/'stages.json').read_text())}
        for file in sorted(directory.glob('*-users_stats.csv')):
            users=int(file.name.split('-')[0]);samples=[r for r in telemetry if r['users']==users]
            failures=list(csv.DictReader((directory/f'{users}-users_failures.csv').open()))
            rejected=sum(int(r['Occurrences']) for r in failures if 'HTTP 503' in r['Error'])
            for row in csv.DictReader(file.open()):
                if row['Name']!='Aggregated':continue
                total=int(row['Request Count']);failed=int(row['Failure Count'])
                output.append({'scenario':directory.name,'users':users,'requests':total,'failed':failed,
                               'http_503':rejected,'other_failures':failed-rejected,
                               'request_rate':float(row['Requests/s']),
                               'success_rate':float(row['Requests/s'])-float(row['Failures/s']),
                               'failure_fraction':failed/total if total else 0,
                               'p50_ms':float(row['50%']),'p95_ms':float(row['95%']),'p99_ms':float(row['99%']),
                               'max_idle_in_transaction':max((sum(s['count'] for s in r['states'] if s['state']=='idle in transaction') for r in samples),default=0),
                               'max_idle_transaction_seconds':max((s['max_transaction_seconds'] or 0 for r in samples for s in r['states'] if s['state']=='idle in transaction'),default=0),
                               'max_connections':max((sum(s['count'] for s in r['states']) for r in samples),default=0),
                               'recovery_status':probes.get(users,{}).get('status'),
                               'recovery_seconds':probes.get(users,{}).get('seconds')})
    return sorted(output,key=lambda r:(r['scenario'],r['users']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    rows=summarize(args.root)
    (args.root/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    if rows:
        with (args.root/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))
