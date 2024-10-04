all: run test

run:
	cmake -H. -Bbuild -DCMAKE_PREFIX_PATH="$(python -c 'from torch.utils import cmake_prefix_path as _cpp; print(_cpp)')"
	cmake --build ./build

test:
	python time.py
	@echo "========================================"
	python model.py

clean:
	rm -rf build loss.png
