import sys
import matplotlib.pyplot as plt
import pickle
import os


def list_average(a, rnd=True):
    avg = sum(a) / len(a)
    avg = avg * 100
    if rnd:
        avg = round(avg, 1)
    
    return avg
    
        


def main():   
    if len(sys.argv) < 3:
        print(f"Usage: python3 {__file__} <train_metrics_pickle_file> <save_graphs (0/1)>")
        sys.exit(-1)
        
        
    pickle_file_path = os.path.abspath(sys.argv[1])
    save_graphs = int(sys.argv[2])
    f = open(pickle_file_path, 'rb')
    hist = pickle.load(f)
    f.close()
    base_savename = None
    
    if save_graphs:
        base_savename = pickle_file_path.split(".")
        
        if len(base_savename) > 2:
            raise ValueError(f"Too many dots ({len(base_savename)-1}) in pickle file path (max 1)")
        
        base_savename = base_savename[0]
    
    
    single_output_model_visualise(hist, base_savename=base_savename)
    
    


def single_output_model_visualise(hist, base_savename=None):   
    
    # print(f"Average val_accuracy = {list_average(hist['val_accuracy'], True)}")
    
    
    plt.plot(hist['accuracy'])
    plt.plot(hist['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    
    if base_savename:
        graph_folder_name = os.path.join(os.path.dirname(base_savename), 
                                             "graphs", os.path.basename(base_savename))
        
        os.makedirs(graph_folder_name, exist_ok=True)
        graph_file_name = f"accuracy_graph.jpg"
        graph_file_path = os.path.join(graph_folder_name, graph_file_name)
        plt.savefig(graph_file_path)
    
    plt.show()   
    
    
    plt.plot(hist['loss'])
    plt.plot(hist['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')

    if base_savename:
        graph_file_name = f"loss_graph.jpg"
        graph_file_path = os.path.join(graph_folder_name, graph_file_name)
        plt.savefig(graph_file_path)
        
    plt.show()         
        
        
        
        

if __name__ == "__main__":
    main()
   
        