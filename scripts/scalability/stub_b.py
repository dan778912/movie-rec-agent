"""Deterministic HTTP latency fixture, not a recommendation model."""
import asyncio
import os
from fastapi import FastAPI
app = FastAPI()
@app.post('/decide')
async def decide():
    await asyncio.sleep(float(os.environ.get('BENCH_B_DELAY', '0.25')))
    return {'action': 'ask', 'question': 'Which genre would you like?'}
