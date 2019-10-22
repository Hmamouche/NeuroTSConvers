#python multi_subjects.py -s 2 3 5 6 7 8 9 10 11 13 15 17 18 20 21 22 23  -rg 1 2 3 4 -p -5 -rm -t pred #-crossv


#python multi_subjects.py -s 2 3 5 6 7 8 9 10 11 13 15 17 18 20 21 22 23 -rg 5 6 7 8 9 -p 6 -rm -t pred

#python multi_subjects.py -s 2 3 5 9 15 17 18 20 -rg  5 -p 6 -rm -t pred
#python multi_subjects.py -s 2 3 5 9 15 17 18 20 -rg  5 6 7 8 9 -p 5 -t pred -rm
python multi_subjects.py -s 2 3 5 6 7 8 9 10 11 13 15 17 18 20 21 22 23 -rg 5 6 7 8 9 -p 5 -t pred -lstm
