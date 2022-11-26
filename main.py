import argparse
from train import Train
from utils.data_prep import DataPreparation
from utils.report import ReportAccuracies

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, default = './', help='Directory where the data is stored')
parser.add_argument('--epochs', type=int, default = 150, help='Number of Epochs of training')
parser.add_argument('--folds', type=int, default = 10, help='Number of Folds of training')
parser.add_argument('--batch_size', type=int, default = 192, help='Batch size for training')
parser.add_argument('--learning_rate', type=float, default = 0.001, help='Initial Learning Rate')
args = parser.parse_args()

def run(data_input_file):
  X,Y,folds,= DataPreparation(data_input_file)
  avg_acc, avg_recall,avg_f1= Train(X,Y,folds=args.folds,batch_size=args.batch_size,epochs=args.epochs,learning_rate=args.learning_rate)
  ReportAccuracies(avg_acc, avg_recall,avg_f1 )

if __name__ == '__main__':
    data_input_file = args.data_directory + 'data.npz'
    run(data_input_file)