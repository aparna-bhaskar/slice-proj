#!/usr/bin/env python3
"""Verify final RTL stores against the independently generated PyTorch reference."""
import argparse
import json
import math
import sqlite3
import struct
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('trace')
p.add_argument('elf')
p.add_argument('reference')
p.add_argument('--nm', default='llvm-nm')
a = p.parse_args()
r = json.load(open(a.reference))
symbols = subprocess.check_output([a.nm, a.elf], text=True)
base = int(next(line.split()[0] for line in symbols.splitlines()
                if line.endswith(' Y_raw')), 16)
count = len(r['expected'])
memory = bytearray(4 * count)
written = set()
with sqlite3.connect(a.trace) as db:
    for addr, size, value in db.execute(
            'SELECT address,size,data FROM dmem WHERE store=1 ORDER BY id'):
        for offset, byte in enumerate((value & ((1 << (8 * size)) - 1)).to_bytes(size, 'little')):
            index = addr + offset - base
            if 0 <= index < len(memory):
                memory[index] = byte
                written.add(index)
assert len(written) == len(memory), 'Missing output stores'
got = struct.unpack('<' + 'f' * count, memory)
assert all(math.isfinite(x) for x in got), 'Nonfinite output'
errors = [abs(x - y) for x, y in zip(got, r['expected'])]
assert max(errors) <= r['atol'], f'Max absolute error: {max(errors)}'
seq = r['seq']
row_error = 0.0
for row in range(r['heads'] * seq):
    query = row % seq
    start = max(0, query + 1 - r['window'])
    values = got[row * seq:(row + 1) * seq]
    assert all(x == 0.0 for j, x in enumerate(values) if j < start or j > query), 'Mask mismatch'
    row_error = max(row_error, abs(sum(values) - 1.0))
assert row_error <= r['atol'], f'Row normalization error: {row_error}'
print(json.dumps({'result': 'PASS', 'outputs': count, 'heads': r['heads'],
                  'seq': seq, 'window': r['window'], 'max_abs_error': max(errors),
                  'atol': r['atol'], 'max_row_sum_error': row_error}, indent=2))
