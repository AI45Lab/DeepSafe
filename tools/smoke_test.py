import sys
import os

print("Starting smoke test...")
print("Testing datasets...")
import uni_eval.datasets
print("Testing evaluators...")
import uni_eval.evaluators
print("Testing metrics...")
import uni_eval.metrics
print("Testing summarizers...")
import uni_eval.summarizers
print("All modules imported successfully!")
