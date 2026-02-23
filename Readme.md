# My First Neural Network: A Learning Journey

**MNIST Handwritten Digit Classifier**  
_Built while learning Andrew Ng's Advanced Learning Algorithms course_

---

## Why I Built This

I didn't set out to build the "perfect" MNIST classifier. I was simply curious.

After going through Andrew Ng's Advanced Learning Algorithms course, I had all this theory in my head about neural networks, backpropagation, activation functions, and loss functions. But I hadn't actually _built_ anything yet. I wanted to see if I could take what I learned and turn it into something real.

So I decided to build my first neural network from scratch.

This README isn't a polished portfolio piece. It's the story of what I learned, what confused me, what surprised me, and how I figured things out along the way.

---

## Table of Contents

- [The Starting Point](#the-starting-point)
- [First Obstacles](#first-obstacles)
- [The "Aha" Moments](#the-aha-moments)
- [What I Built](#what-i-built)
- [The Results](#the-results)
- [What I Learned Beyond the Course](#what-i-learned-beyond-the-course)
- [What Surprised Me](#what-surprised-me)
- [What Still Confuses Me](#what-still-confuses-me)
- [If I Could Do It Again](#if-i-could-do-it-again)
- [What's Next](#whats-next)
- [Technical Details](#technical-details)
- [How to Run This Project](#how-to-run-this-project)

---

## The Starting Point

Before this project, I was just someone who had watched lecture videos and taken notes. I understood the concepts in theory:

- Forward propagation
- Activation functions
- Loss functions
- Gradient descent

But I hadn't actually _implemented_ any of it.

I also had exams coming up, so I didn't have unlimited time to experiment. I needed to focus, build something that worked, and learn as much as I could in the process.

Oh, and I had just gotten a new laptop. Part of me was excited to see how fast it could train a neural network. (Spoiler: 20 seconds for 10 epochs. I was impressed.)

---

## First Obstacles

### Obstacle #1: "Wait, how do I even SEE the data?"

This sounds basic now, but my very first problem was: _How do I look at the MNIST images?_

I had loaded the dataset using Keras, but I had this array of numbers and no idea how to actually visualize it. I Googled around and discovered I could use `matplotlib` to display the images. That was my first small win.

### Obstacle #2: "What does 'normalize by 255' even mean?"

Andrew's course mentioned normalization, but I didn't fully understand _why_ I was dividing pixel values by 255.

I knew pixel values ranged from 0-256 (grayscale), but why divide by that specific number? Through searching and asking questions, I learned it was about scaling the inputs to [0, 1] so gradient descent could converge faster. That took time to sink in.

### Obstacle #3: "My data is 28×28, but Dense layers want 1D input?"

This one confused me for a while. The MNIST images are 28×28 grids, but fully connected (Dense) layers expect a 1D vector. How do I bridge that gap?

That's when I discovered the **Flatten layer** — something that wasn't covered in Andrew's course. It was my first time learning something _beyond_ the lectures by necessity.

And while researching Flatten, I stumbled upon **Dropout** too. Another layer Andrew hadn't covered in detail. I read about it, understood it helps prevent overfitting by randomly "turning off" neurons during training, and decided to add it to my model.

These weren't just copy-paste decisions. I had to understand _why_ they existed and _when_ to use them.

---

## The "Aha" Moments

### Moment #1: Seeing the Greyscale Pixels

When I first visualized the MNIST images, I was struck by how the pixel intensity created the digit shapes. You could literally see lighter and darker regions forming numbers. It made me realize: the neural network is just learning patterns in these pixel intensities.

That simple visualization made the whole problem feel more concrete.

### Moment #2: The Model is LEARNING

The first time I hit "run" and watched the training loop, I was genuinely excited.

```
Epoch 1/10
loss: 0.2543 - accuracy: 0.9234 - val_loss: 0.1234 - val_accuracy: 0.9623
```

The loss was going DOWN. The accuracy was going UP.

It was actually working.

I had heard that neural networks take time to train, so I was watching both the progress bar and my laptop's performance. Two things were happening at once:

1. My first neural network was learning
2. My new laptop was handling it smoothly

I felt like I had crossed a threshold from "someone who watches videos" to "someone who builds models."

### Moment #3: 97.46% Accuracy

When I saw the final test accuracy, my first thought was: _"Is that good?"_

Then I remembered Andrew's advice: always compare to a baseline. For MNIST, human-level performance is around 98%. My model got 97.46%.

That's when it hit me — this simple 3-layer network I built was performing nearly as well as humans on this task.

I was satisfied. But I also wanted to see _where_ it failed.

---

## What I Built

### The Architecture

```
Input: 28×28 grayscale images (784 pixels)
         ↓
    Flatten Layer (convert 2D → 1D)
         ↓
Dense Layer 1: 128 neurons, ReLU activation
         ↓
Dropout: 20% (prevent overfitting)
         ↓
Dense Layer 2: 64 neurons, ReLU activation
         ↓
Dropout: 20%
         ↓
Output Layer: 10 neurons, Softmax activation
         ↓
Predicted digit (0-9)
```

**Why these choices?**

Honestly? I didn't experiment much because I had exams coming up. I went with common conventions:

- **128 and 64 neurons**: Powers of 2, common starting points, creates a "funnel" effect
- **ReLU activation**: Standard for hidden layers, avoids vanishing gradients
- **Dropout 20%**: Prevents overfitting without being too aggressive
- **Softmax output**: Converts raw scores to probabilities for 10 classes

If I had more time, I would have experimented with different architectures. But this worked, so I moved forward.

### The Code

Here's what surprised me: **building the model only took 12 lines of code**.

```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(10, activation='softmax')
])
```

That's it. TensorFlow/Keras made it shockingly simple.

The _hard_ part wasn't building the model. It was:

- Understanding what each piece does
- Preprocessing the data correctly
- Interpreting the results
- Visualizing everything (that took WAY longer than I expected)

---

## The Results

### Overall Performance

| Metric            | Value        |
| ----------------- | ------------ |
| **Test Accuracy** | **97.46%**   |
| **Training Time** | 20.3 seconds |
| **Parameters**    | 109,386      |
| **Epochs**        | 10           |

### What the Confusion Matrix Taught Me

When I looked at which digits the model confused, most of the mistakes made sense:

- **4 ↔ 9**: They can look similar with certain handwriting styles
- **3 ↔ 5**: Rounded shapes overlap
- **7 ↔ 1**: Vertical strokes can be ambiguous

I looked at the actual misclassified images, and honestly? Even I would have struggled with some of them. The handwriting was genuinely ambiguous.

That gave me confidence that the model wasn't just randomly guessing — it was making the same kinds of mistakes a human might make.

### Per-Digit Performance

Some digits were easier than others:

- **Best**: Digit 1 (99.12%) — makes sense, it's just a vertical line
- **Worst**: Digit 8 (95.83%) — complex curves that overlap with other digits

---

## What I Learned Beyond the Course

Andrew Ng's course gave me the fundamentals, but building this project taught me things the lectures didn't cover:

### 1. **Flatten and Dropout Layers**

These weren't in the course, but I needed them. I had to research, understand, and implement them myself.

### 2. **The Importance of Data Visualization**

Before building anything, I spent time _looking_ at the data. Visualizing samples, checking distributions, understanding the problem visually. That wasn't emphasized enough in lectures.

### 3. **Normalization Isn't Just a Formula**

I memorized "divide by 255" but didn't understand _why_ until I implemented it. Now I get it: it's about helping gradient descent converge faster.

### 4. **Building Feels Different Than Watching**

Watching Andrew explain backpropagation vs. actually running `model.fit()` and seeing loss decrease — those are completely different experiences. One is abstract, the other is tangible.

### 5. **Debugging Is Part of Learning**

I hit errors. Missing commas, wrong shapes, mismatched dimensions. Each bug taught me to pay closer attention to details.

---

## What Surprised Me

### Surprise #1: How EASY Model Building Was

12 lines of code. That's all it took to define the architecture. TensorFlow abstracted away so much complexity. The _understanding_ is hard, but the _implementation_ is surprisingly straightforward.

### Surprise #2: How LONG Visualization Took

Writing code to plot training curves, confusion matrices, and sample predictions took forever. Way longer than building the model itself.

### Surprise #3: The Model Worked on the First Try

I expected bugs, errors, failed training runs. But once I got the data preprocessed correctly, the model just... worked. First try. 97.46% accuracy.

That taught me: if you understand the fundamentals, modern tools make the implementation surprisingly forgiving.

---

## What Still Confuses Me

I'm going to be honest here: **I don't fully understand how everything works without TensorFlow.**

I can _use_ TensorFlow to build models. But if you asked me to implement backpropagation from scratch using just NumPy? I'd struggle.

I understand the _concepts_:

- Forward pass: multiply inputs by weights, add biases, apply activations
- Loss calculation: compare predictions to true labels
- Backward pass: calculate gradients of loss with respect to weights
- Weight update: adjust weights in the direction that reduces loss

But I haven't _done_ it by hand. That's my next challenge.

I want to understand the "under the hood" mechanics — the raw matrix operations, the gradient calculations, the weight updates — without relying on TensorFlow to handle it for me.

---

## If I Could Do It Again

If I could go back and give advice to myself before starting this project, I'd say:

**"Experiment more. Try building parts of it manually without TensorFlow."**

I focused on getting it working quickly because of time constraints (exams). That was practical, but I missed an opportunity to go deeper.

Next time, I want to:

- Implement a simple neural network from scratch using just NumPy
- Manually calculate gradients for at least one layer
- Build custom activation functions to see how they work
- Experiment with different architectures and hyperparameters

Speed was necessary this time. Depth will be my focus next time.

---

## What's Next

This project was built on a "toy" dataset — MNIST is clean, well-formatted, and has been solved a thousand times.

**My next goal: work with a real-world dataset.**

Not Titanic. Not Iris. Not another Kaggle competition classic.

I want to find messy, real data with missing values, imbalanced classes, and ambiguous features. The kind of data you'd actually encounter in industry.

I want to see how the techniques I learned here hold up when the data isn't perfect.

---

## The Biggest Lesson

Looking back at the entire journey — from starting Andrew Ng's course to finishing this project — the biggest lesson I learned wasn't technical.

It was this:

**Showing up every day is the key to advancing.**

Some days I only had 30 minutes. Some days I was tired. Some days I didn't feel like I was making progress.

But I showed up. I opened the notebook. I wrote a few lines of code. I fixed one bug. I read one article.

And those small, consistent efforts compounded into this finished project.

That's the lesson I'll carry forward into every future project.

---

## What I Want You to Know

If you're a recruiter or hiring manager reading this, here's what I want you to take away about who I am:

**I'm curious and go beyond what's taught.**  
I didn't just follow Andrew Ng's course step-by-step. When I hit roadblocks (like needing the Flatten layer), I researched, learned, and implemented solutions on my own. I'm the kind of person who asks "why" and doesn't stop at "it works."

**I'm honest about what I know and don't know.**  
I can build a neural network with TensorFlow. But I openly admit I don't fully understand how backpropagation works from scratch yet. That self-awareness means I'm coachable, I know what gaps to fill, and I won't pretend to know more than I do.

I'm not the most experienced ML engineer. But I'm someone who will keep learning, keep asking questions, and keep building.

---

## Technical Details

For those who want the specifics:

### Dataset

- **Source**: MNIST (built into Keras)
- **Training samples**: 60,000 (split 80/20 for train/validation)
- **Test samples**: 10,000
- **Image size**: 28×28 grayscale pixels
- **Classes**: 10 (digits 0-9)

### Preprocessing

- Normalized pixel values from [0, 255] to [0.0, 1.0]
- One-hot encoded labels for categorical crossentropy loss
- No data augmentation (rotation, scaling, etc.)

### Model Architecture

- **Input**: 784 features (28×28 flattened)
- **Hidden Layer 1**: 128 neurons, ReLU activation
- **Dropout**: 20%
- **Hidden Layer 2**: 64 neurons, ReLU activation
- **Dropout**: 20%
- **Output**: 10 neurons, Softmax activation
- **Total Parameters**: 109,386

### Training Configuration

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam (learning rate: 0.001)
- **Batch Size**: 32
- **Epochs**: 10
- **Validation Split**: 20%

### Results

- **Test Accuracy**: 97.46%
- **Test Loss**: 0.0834
- **Training Time**: 20.3 seconds
- **Training/Validation Gap**: <2% (no overfitting)

### Performance by Digit

| Digit | Accuracy       |
| ----- | -------------- |
| 0     | 98.57%         |
| 1     | 99.12% ★ Best  |
| 2     | 96.51%         |
| 3     | 97.03%         |
| 4     | 96.95%         |
| 5     | 97.08%         |
| 6     | 98.02%         |
| 7     | 96.79%         |
| 8     | 95.83% ★ Worst |
| 9     | 96.63%         |

---

## How to Run This Project

### Prerequisites

```bash
Python 3.8+
TensorFlow 2.x
NumPy
Matplotlib
Seaborn
scikit-learn
```

### Installation

1. Clone the repository

```bash
git clone https://github.com/yourusername/mnist-digit-classifier.git
cd mnist-digit-classifier
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the notebook

```bash
jupyter notebook digit_classifier_model.ipynb
```

### requirements.txt

```
tensorflow>=2.10.0
numpy>=1.23.0
matplotlib>=3.5.0
seaborn>=0.12.0
scikit-learn>=1.1.0
jupyter>=1.0.0
```

---

## Acknowledgments

**Andrew Ng** — Your Advanced Learning Algorithms course gave me the foundation to build this. Thank you for making deep learning accessible.

**The MNIST Dataset** — Created by Yann LeCun, Corinna Cortes, and Christopher J.C. Burges. Still the perfect first project for learning neural networks.

**Stack Overflow & AI Assistants** — For helping me debug, understand, and implement concepts I was stuck on.

**My New Laptop** — For handling this training run in 20 seconds. We're going to build a lot more together.

---

## Final Thoughts

This was my first neural network. It's not perfect. There's a lot I still don't understand.

But two months ago, I didn't know what a Dense layer was. Now I've built a working classifier that achieves near-human performance on a real task.

That's progress.

And I'm just getting started.

---

**Project Status**: ✅ Complete  
**What I'm Building Next**: A project with real-world, messy data  
**What I'm Learning Next**: How to implement neural networks from scratch without TensorFlow

---

<div align="center">

_Built with curiosity, persistence, and a lot of Googling_  
_Learning in public 🚀_

</div>
