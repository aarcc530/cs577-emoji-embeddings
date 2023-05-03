import pandas as pd
import emoji
import argparse

parser = argparse.ArgumentParser(
                    prog='File converter',
                    description='Convert CBOW files into proper GloVe/Word2Vec format')

parser.add_argument('-f', '--filename')
args = parser.parse_args()
assert args.filename


embeddings = pd.read_csv(args.filename)
embeddings[0] = embeddings.apply(lambda row: emoji.demojize(row[0], language='alias'), axis=1)
embeddings.drop(embeddings.tail(1).index, inplace=True)
embeddings['numbers'] = embeddings.apply(lambda row: [float(x[:-1]) for x in row['Embedding'][1:-1].split()], axis=1)
numbers = pd.DataFrame(embeddings['numbers'].tolist(), columns=range(1, 51))
numbers.insert(0, 'emoji', embeddings.iloc[:, 0])
numbers.to_csv(f'{args.filename[:-4]}.txt', sep=' ', header=False, index=False)
with open(f'{args.filename[:-4]}.txt', "r+") as f: s = f.read(); f.seek(0); f.write(f"{len(numbers.index)} 50\n" + s)