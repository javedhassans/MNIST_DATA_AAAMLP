import config
import engine
import model
import dataset

for t in range(config.epochs):
    print(f"Epoch {t+1} \n ___________________________")
    engine.train(dataset.train_dataloader, model.model, engine.loss_fn, engine.optimizer)
    engine.test(dataset.test_dataloader, model.model, engine.loss_fn)

print("Done")