all: run

run:
	python time.py > test_time.log
	@echo "========================================"
	python model.py

clean:
	rm -rf test_time.log loss.png
