python3 train.py -classes 6 -custom_preprocess 0 -network inceptionv4 -epochs 305

echo "First completed (3rd batch)" >> logger.txt

python3 train.py -classes 6 -custom_preprocess 1 -network inceptionv4 -epochs 306

echo "Second completed!" >> logger.txt

python3 train.py -classes 6 -custom_preprocess 0 -network nasnet -epochs 307

echo "Third completed!" >> logger.txt


python3 train.py -classes 6 -custom_preprocess 1 -network nasnet -epochs 308

echo "Fourth completed!" >> logger.txt
