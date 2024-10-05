CMAKE_PREFIX_PATH := $(shell python -c 'from torch.utils import cmake_prefix_path as _cpp; print(_cpp)')

all: run test

run:
	cmake -H. -Bbuild -DCMAKE_PREFIX_PATH="$(CMAKE_PREFIX_PATH)"
	cmake --build ./build

test:
	python time.py > test_time.log
	@echo "========================================"
	python model.py

clean:
	rm -rf build test_time.log loss.png
