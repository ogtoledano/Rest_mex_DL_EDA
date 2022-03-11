from utils.preprocess_emo_eval_es_mt5 import build_dataset_and_dict
from simpletransformers.t5 import T5Model, T5Args

if __name__ == "__main__":
    train,test,eval = build_dataset_and_dict()

    model_args = T5Args()
    model_args.max_seq_length = 196
    model_args.train_batch_size = 8
    model_args.eval_batch_size = 8
    model_args.num_train_epochs = 1
    model_args.evaluate_during_training = False
    model_args.use_multiprocessing = False
    model_args.fp16 = False
    model_args.save_steps = -1
    model_args.save_eval_checkpoints = False
    model_args.save_model_every_epoch = False
    model_args.no_cache = True
    model_args.reprocess_input_data = True
    model_args.overwrite_output_dir = True
    model_args.num_return_sequences = 1
    model_args.wandb_project = "MT5 mixed tasks"

    model = T5Model("mt5", "google/mt5-base", args=model_args)

    # Train the model
    model.train_model(train.astype(str), eval_data=eval.astype(str))

    # Optional: Evaluate the model. We'll test it properly anyway.
    results = model.eval_model(test.astype(str), verbose=True)
