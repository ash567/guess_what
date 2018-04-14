# QGen config

The qgen task requires to produce a new question given an image and the history of dialogues.

The configuration file is divided into three parts:
 - QGen model definition
 - QGen training
 - Others

The keyword "model" refers model configuration of the qgen:
```
"model": {

    "word_embedding_size": int,         # dimension of the word embedding for the dialogue
    "num_lstm_units": int,              # dimension of the LSTM for the dialogue
    "image_embedding_size": int,        # dimension of the image projection
    "attention": string                 # options: none, mean classic and glimpse

    "image": {                          # configuration of the inout image
      "image_input": "features"/"raw",  # select the image inputs: raw vs feature
      "dim": list(int)                  # provide the image input dimension
    }

  },
```

The "optimizer" key refers to the training hyperparameters:


```
  "optimizer": {
    "no_epoch": int,            # the number of traiing epoches
    "learning_rate": float,     # Adam initial learning rate
    "batch_size": int,          # training batch size
    "clip_val": int             # gradient clip to avoid RNN gradient explosion
  },
 ```

Other parameters can be set such as:

```
  "seed": -1                                       # define the training seed; -1 -> random seed
 ```

Eg: 


{

  "model": {

    "word_embedding_size": 512,
    "num_lstm_units": 1024,
    "image_embedding_size": 512,
    "attention" : {
        "mode": "none",
        "no_attention_mlp": 256,
        "no_glimpses": 2
      },

    "image": {
      "image_input": "raw",
      "dim": [224, 224, 3]
    }

  },

  "optimizer": {
    "no_epoch": 30,
    "learning_rate": 1e-4,
    "batch_size": 32,
    "clip_val": 5
  },


  "seed": -1,
  "freq": 1,
  "comments":{
    "attentiong": "Attention inserted",
    "dimetiontion of image beforee": "if using raw take 224, 224, 3 else just 1000"

  }
}


------------------------------------------------ ---------------------------------------------------------

 Merge with 

 # Guesser config

The guesser task requires to select an object within a list of objects object given a dialogue and a image.

The configuration file is divided into three parts:
 - Guesser model definition
 - Guesser training
 - Others

In this model, objects are encoded by there spatial information and their category.

The keyword "model" refers to the model architecture of the guesser:
```
"model": {

    "word_emb_dim": 512,   # dimension of the word embedding for the dialogue
    "num_lstm_units": 512, # dimension of the LSTM for the dialogue

    "cat_emb_dim": 256,    # dimension of the object category embedding
    "no_categories": 90    # number of object category (90 for MS coco)
    "spat_dim": 8,         # dimension of the spatial information
    "obj_mlp_units": 512,  # number of hidden units to build the full object embedding

    "dialog_emb_dim": 512, # Projection size for the dialogue and the objects
  },
```

The keyword "optimizer" key refers to the training hyperparameters:


```
  "optimizer": {
    "no_epoch": int,            # the number of traiing epoches
    "learning_rate": float,     # Adam initial learning rate
    "batch_size": int,          # training batch size
    "clip_val": int             # gradient clip to avoid RNN gradient explosion
  },
 ```

Other parameters can be set such as:

```
  "seed": -1                                       # define the training seed; -1 -> random seed
 ```


 Eg: 

 {
  "model": {
    "word_emb_dim": 512,
    "num_lstm_units": 512,
    "cat_emb_dim": 256,
    "obj_mlp_units": 512,
    "dialog_emb_dim": 512,
    "spat_dim": 8,
    "no_categories": 90
  },

  "optimizer": {
    "no_epoch": 20,
    "learning_rate": 1e-4,
    "batch_size": 64,
    "clip_val": 5
  },

  "seed": -1

}
