from data import *
import torch.nn as nn
from model import *
import matplotlib.pyplot as plt

def train(
    train_set, 
    batch_size, 
    shuffle, 
    lr, 
    num_epochs,
    generator,
    discriminator
):
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle
    )

    # Loss function
    loss_function = nn.BCELoss()

    # Optimizer init
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            # Data for training the discriminator
            real_samples_labels = torch.ones((batch_size, 1))
            latent_space_samples = torch.randn((batch_size, 2))
            generated_samples = generator(latent_space_samples)
            generated_samples_labels = torch.zeros((batch_size, 1))
            all_samples = torch.cat((real_samples, generated_samples))
            all_samples_labels = torch.cat(
                (real_samples_labels, generated_samples_labels)
            )

            # Training the discriminator
            discriminator.zero_grad()
            output_discriminator = discriminator(all_samples)
            loss_discriminator = loss_function(
                output_discriminator, all_samples_labels)
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # Data for training the generator
            latent_space_samples = torch.randn((batch_size, 2))

            # Training the generator
            generator.zero_grad()
            generated_samples = generator(latent_space_samples)
            output_discriminator_generated = discriminator(generated_samples)
            loss_generator = loss_function(
                output_discriminator_generated, real_samples_labels
            )
            loss_generator.backward()
            optimizer_generator.step()

            # Show loss
            if epoch % 10 == 0 and n == batch_size - 1:
                print(f"Epoch: {epoch} Loss D.: {loss_discriminator}")
                print(f"Epoch: {epoch} Loss G.: {loss_generator}")


def evaluate(latent_space_samples, generator):
    generated_samples = generator(latent_space_samples)
    generated_samples = generated_samples.detach()
    plt.plot(generated_samples[:, 0], generated_samples[:, 1], ".")
    plt.show()

if __name__ == "__main__":
    train_data, train_set = get_traindata(data_length=1024)

    # viz(train_data) # Visualize the data

    # Hyperparameters
    batch_size = 32
    shuffle = True
    lr = 0.001
    num_epochs = 300
    
    # Model init
    generator = Generator()
    discriminator = Discriminator()

    # Evaluate param
    latent_space_samples = torch.randn(100, 2)

    train(
        train_set=train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        lr=lr,
        num_epochs=num_epochs,
        generator=generator,
        discriminator=discriminator
    )

    evaluate(
        latent_space_samples=latent_space_samples,
        generator=generator
    )

    
    

