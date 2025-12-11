# Makefile for pattingalloang

.PHONY: install dev test clean run-all run-case1 run-case2 run-case3 run-case4 run-case5 run-case6 run-case7 run-case8 help

help:
	@echo "pattingalloang - Aizawa Attractor Analysis"
	@echo ""
	@echo "Installation:"
	@echo "  make install     - Install package with pip"
	@echo "  make dev         - Install in development mode with poetry"
	@echo ""
	@echo "Testing:"
	@echo "  make test        - Run all tests"
	@echo ""
	@echo "Running simulations:"
	@echo "  make run-all     - Run all test cases"
	@echo "  make run-case1   - Standard Aizawa (quick)"
	@echo "  make run-case2   - High resolution"
	@echo "  make run-case3   - Long trajectory (chaos metrics)"
	@echo "  make run-case4   - Parameter variation"
	@echo "  make run-case5   - Multi-trajectory"
	@echo "  make run-case6   - Butterfly wings"
	@echo "  make run-case7   - Chaotic spiral"
	@echo "  make run-case8   - Double loop"
	@echo ""
	@echo "GPU acceleration:"
	@echo "  make run-case1-gpu - Run case1 with GPU"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean       - Remove outputs and cache"

install:
	pip install .

dev:
	poetry install

test:
	pytest tests/ -v

run-all:
	pattingalloang --all

run-case1:
	pattingalloang case1

run-case2:
	pattingalloang case2

run-case3:
	pattingalloang case3

run-case4:
	pattingalloang case4

run-case5:
	pattingalloang case5

run-case6:
	pattingalloang --config configs/case6_butterfly_wings.txt

run-case7:
	pattingalloang --config configs/case7_chaotic_spiral.txt

run-case8:
	pattingalloang --config configs/case8_double_loop.txt

run-case1-gpu:
	pattingalloang case1 --gpu

clean:
	rm -rf outputs/
	rm -rf logs/
	rm -rf __pycache__/
	rm -rf src/pattingalloang/__pycache__/
	rm -rf src/pattingalloang/core/__pycache__/
	rm -rf src/pattingalloang/io/__pycache__/
	rm -rf src/pattingalloang/utils/__pycache__/
	rm -rf src/pattingalloang/visualization/__pycache__/
	rm -rf .pytest_cache/
	rm -rf *.egg-info/
	rm -rf dist/
	rm -rf build/
