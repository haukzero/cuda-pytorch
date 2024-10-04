all: run

install:
	python setup.py install

run:
	python time.py
	@echo "========================================"
	python model.py

clean:
	rm -rf loss.png

clear:
	pip uninstall add2
	rm -rf build add2.egg-info dist loss.png

