all: run

install:
	python setup.py install

run:
	python time.py > test_time.log
	@echo "========================================"
	python model.py

clean:
	rm -rf loss.png

clear:
	pip uninstall add2
	rm -rf build add2.egg-info dist test_time.log loss.png

