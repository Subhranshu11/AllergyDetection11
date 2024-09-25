from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1./255,
    validation_split=0.2  # Set 20% of the data for validation
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
# Apply augmentation to the dataset
#train_datagen = datagen.flow_from_directory(r'/content/drive/MyDrive/Food Images', target_size=(256, 256), batch_size=32, class_mode='binary')

train_data_path = 'Food Images\Food Images'
val_data_path = 'Food Images\Valdata'
#train_images = train_images / 255.0
# Load training data
train_generator = datagen.flow_from_directory(
    train_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Use the training subset
)
# Load validation data
val_generator = val_datagen.flow_from_directory(
    train_data_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Use the validation subset
)

#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize

#nltk.download('stopwords')
#stop_words = set(stopwords.words('english'))

#def preprocess_ingredients(ingredients):
    # Tokenize
 #   word_tokens = word_tokenize(ingredients)
    # Remove stopwords
  #  filtered_ingredients = [w for w in word_tokens if not w.lower() in stop_words]
   # return filtered_ingredients

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# Add Convolutional Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more Conv layers
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Dense Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
# Change the number of neurons in the final Dense layer to match the number of classes
num_classes = train_generator.num_classes  # Automatically gets the number of classes from the generator
model.add(Dense(num_classes, activation='softmax'))  # Output layer with the number of classes

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, concatenate


history = model.fit(train_generator, epochs=10, validation_data=val_generator)

val_loss, val_acc = model.evaluate(val_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_acc}')