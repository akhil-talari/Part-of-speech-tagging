Username: powerstar

How to compile the source code:
***********WINDOWS*************
To run python files on cmd:
* open cmd
* cd to path where you copied all the input datasets and 2 python files
* python <filename.py> =>to run the python file

I have provided 2 python files:
1. main.py
2. accuracy.py

Outputs when you run above python files:
1.	main.py =>
One output file is created.
a."submission.txt", this contains the POS predicted taggers run on test.conll & dev.conll
---------------------------------------------------------------------
2.	accuracy.py => On console, accuracy on dev set is displayed.
Two output files are created.
a.	"dev_test_file.txt", this contains only the 1st & 2nd column of dev.conll
b.	"dev-output.txt", this contains the POS predicted taggers run on dev_test_file.txt
On comparing “dev-output.txt” with “dev.conll”, accuracy is predicted
----------------------------------------------------------------------
Note:
1. The input datasets "train.conll", "test.conll", "dev.conll" should be in the same directory where, the python source codes are copied.
(this should be easiest way to run the program quick.)

2. Every time you run the code (main.py):
On console: processing...
		 finished!!! 
will be displayed. It takes approx 2 minutes to get finished.

3. All the output files are created in the same directory where the source code is present.
