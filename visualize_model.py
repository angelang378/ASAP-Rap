from torch.nn import Linear

from music_data import *
from artist_model import *
from torchviz import make_dot

def visualize_trained_artist_model(data, mpath):
    song_titles, song_data = get_input_output(data, user_1="Taylor Swift", user_2="Kanye West")

    scaler = StandardScaler()
    predict_tensor = scaler.fit_transform(
        torch.from_numpy(song_data.values))

    # Loading the saved model
    model = BinaryClassification(predict_tensor.shape[1])
    model.load_state_dict(torch.load(mpath))

    prediction = model(torch.from_numpy(predict_tensor).float())  # Give dummy batch to forward().

    make_dot(prediction, params=dict(list(model.named_parameters()))).render("rnn_torchviz1", format="png")

    input_weights = model.layer_1.weight.data
    x_np = input_weights.numpy()
    x_df = pd.DataFrame(x_np)
    x_df.to_csv('weights1.csv')

def main():
    visualize_trained_artist_model("data/4a6u0ZVG0FWYAJHVggnHAh.csv", "trained_models/trained_model15.pt")
    #visualize_trained_artist_model("data/4a6u0ZVG0FWYAJHVggnHAh.csv", "trained_models/trained_model9.pt")

if __name__ == "__main__":
    main()