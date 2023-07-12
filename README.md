# SimCLR
Trainings

  1. Unsupervised Training:  
Edit ```config.gin``` to change the hyperparameters for ```train.py``` training. Then run ```train.py```. The current settings should yield the best results. (There's one exception: You should increase ```initial_filters``` in ```architecture.gin```, if you have enough VRAM: ```initial_filters = 32``` for 12GB VRAM GPUs, ```initial_filters = 8``` for 6GB VRAM GPUs and ```initial_filters > 32``` for very powerful hardware setups)
  2. Unsupervised AND Semi-Supervised Training every epoch:  
Edit ```ssl_training_config.gin``` to change the hyperparameter. In ```tfds_path = <path_name>``` you HAVE to give the cifar10 version 3.0.2 folder. Otherwise you have to make new cifar10 splits with ```cifar10_pipeline_1%.py```. Then run ```trainSSL.py```. (Same rules for ```initial_filters``` as in ```architecture.gin``` apply)
  3. Unsupervsied Training in all but every N-th Steps and Semi-Supervised Training in the N-th steps:  
Do exactly the same as in 2), but run ```trainSSLeveryN.py``` instead of ```trainSSL.py```.
  
Evaluations

  1. Linear evaluation:  
Requires ```encoder_trainable = False``` in ```config_eval.gin``` and the path to your model checkpoints with ```path_model_id = <path_name>``` in ```evaluation.py```. Then run ```evaluation.py```.
  2. Lower bound:  
Requires ```encoder_trainable = False``` in ```config_eval.gin``` and an empty ```path_model_id = ''``` in ```evaluation.py```.
  3. Upper bound:  
Requires ```encoder_trainable = True``` in ```config_eval.gin``` and an empty ```path_model_id = ''``` in ```evaluation.py```.
  4. Semi-supervised evaluation (finetune for n epochs with only an N% Split of Cifar10, i.e. only 50 images per class with a 1% Split):  
Requires ```encoder_trainable = True``` in ```config_eval.gin``` and the N can be changed via ```gen_pipeline_ssl_eval.useNpercentOfCifar10 = N``` in ```config_eval.gin```. Also define a checkpoints path to ```SSL_eval.py``` with ```path_model_id = <path_name>```. Then run ```SSL_eval.py```.
  5. Semi-supervised lower bound:  
Do the same as in 4), but change ```path_model_id = ''``` in ```SSL_eval.py``` as you don't want to use any checkpoints.
