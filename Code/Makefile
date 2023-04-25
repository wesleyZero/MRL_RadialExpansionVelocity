PYTHON_FILES = REV.py

all:

run:
	python3 $(PYTHON_FILES) > output.txt 

output: 
	2>output_err.txt
	python3 $(PYTHON_FILES) > output.txt
	cat output.txt| grep 'array_function\|videoProcessor\|Processing:' > mod_output.txt && open mod_output.txt

sublime: output.txt
	sublime output.txt

rm:
	rm output.txt
	echo 'remove my poop'


quickPush:
	git add $(PYTHON_FILES)
	git commit -m 'quick save'
	git push origin master 

push:
	git add $(PYTHON_FILES)
	git commit -m '
