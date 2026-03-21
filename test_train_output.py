#!/usr/bin/env python
"""Quick test to verify training output is working."""
import sys
sys.path.insert(0, '.')

print("=" * 70, flush=True)
print("Testing train output with flush=True", flush=True)
print("=" * 70, flush=True)

# Simulate what emit() does
def test_emit(message: str) -> None:
    print(message, flush=True)

for i in range(5):
    test_emit(f"[Test {i+1}] This is a test message with immediate output")

print("=" * 70, flush=True)
print("Output test completed successfully!", flush=True)
print("=" * 70, flush=True)

