"""Offline analysis of nsys SQLite exports; not part of inference serving."""
import collections
import json
import sqlite3
import sys
from pathlib import Path


def analyze(stem):
    report = json.loads(Path(str(stem) + '-smoke.json').read_text())
    connection = sqlite3.connect(str(stem) + '.sqlite')
    connection.row_factory = sqlite3.Row
    rows = [dict(row) for row in connection.execute('''
        SELECT k.start, k.end, k.correlationId, k.graphNodeId, k.deviceId, k.contextId,
               s.value AS name, d.value AS fullName
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName=s.id
        JOIN StringIds d ON k.demangledName=d.id ORDER BY k.start''')]
    starts = [i for i, row in enumerate(rows) if row['name'] == 'embeddingKernel']
    assert len(starts) == len(report['input']) + len(report['output']) - 1
    runtime = {row['correlationId']: dict(row) for row in connection.execute('''
        SELECT r.correlationId, r.start, r.end, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id''')}
    memory_ops = [dict(row) for table in ['CUPTI_ACTIVITY_KIND_MEMCPY', 'CUPTI_ACTIVITY_KIND_MEMSET']
                  for row in connection.execute('SELECT start,end,deviceId,contextId FROM ' + table)]
    decode = []
    own = {'embeddingKernel', 'addKernel', 'normKernel', 'ropeKernel', 'swigluKernel',
           'appendKVKernel', 'attentionKernel', 'argmaxKernel', 'convert'}
    for position in range(len(report['input']), len(starts)):
        chunk = rows[starts[position]:starts[position + 1] if position + 1 < len(starts) else len(rows)]
        totals = collections.defaultdict(float)
        busy = 0
        end = chunk[0]['start']
        for row in chunk:
            busy += max(0, row['end'] - max(end, row['start']))
            end = max(end, row['end'])
            category = row['name'] if row['name'] in own else 'cuBLAS kernels'
            totals[category] += (row['end'] - row['start']) / 1e6
        start, finish = chunk[0]['start'], end
        span = finish - start
        intervals = [(row['start'], row['end']) for row in chunk]
        intervals += [(max(start, row['start']), min(finish, row['end'])) for row in memory_ops
                      if row['deviceId'] == chunk[0]['deviceId'] and row['contextId'] == chunk[0]['contextId']
                      and row['start'] < finish and row['end'] > start]
        device_busy, cursor = 0, start
        for left, right in sorted(intervals):
            device_busy += max(0, right - max(cursor, left))
            cursor = max(cursor, right)
        correlations = {row['correlationId'] for row in chunk}
        api_counts = collections.Counter(runtime[c]['name'] for c in correlations if c in runtime)
        decode.append({'position': position, 'kernels': len(chunk), 'kernelSpanMs': span / 1e6,
                       'kernelBusyMs': busy / 1e6, 'betweenKernelGapMs': (span - busy) / 1e6,
                       'noRecordedDeviceActivityMs': (span - device_busy) / 1e6, 'byCategoryMs': dict(totals), 'correlatedRuntimeCalls': dict(api_counts)})
    assert decode
    mean = lambda key: sum(row[key] for row in decode) / len(decode)
    categories = {key for row in decode for key in row['byCategoryMs']}
    category_means = {key: sum(row['byCategoryMs'].get(key, 0) for row in decode) / len(decode) for key in categories}
    return {'trace': str(stem), 'promptTokens': len(report['input']), 'output': report['output'],
            'decode': decode, 'mean': {key: mean(key) for key in ['kernels', 'kernelSpanMs', 'kernelBusyMs', 'betweenKernelGapMs', 'noRecordedDeviceActivityMs']},
            'meanCategoryMs': category_means,
            'limitations': ['Nsight tracing changes timings; use unprofiled benchmarks for speedup.',
                            'Gaps are within one forward, exclude before/after-forward gaps, and are not SM utilization. No-recorded-device-activity also excludes traced memcpy/memset.',
                            'cuBLAS category is all kernels other than SproutLM named kernels in this trace.']}

results = {Path(stem).name: analyze(Path(stem)) for stem in sys.argv[1:]}
print(json.dumps(results, indent=2))
