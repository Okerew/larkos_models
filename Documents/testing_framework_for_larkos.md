# Testing Framework for larkos

## Test 1: Learning Efficiency

We shall give the model a completely new domain in a dataset, for
example a toy physics world with similar or completely different rules
of physics. It doesn't matter much we basically want to see how much
data is needed before competence appears and how many epochs it takes.
Additionally, we want to observe how quickly loss decreases. We also
want to notice patterns: does it encode the patterns of the physics
world itself, or does it just copy blindly, or accidentally arrive at
the solution? Basically, this needs to be done within a given time.

## Test 2: Domain Transfer

Let's say we give the model some logic puzzles (learning A). We want
then to test how it plans its internal state or how it does
mathematics. We don't expect mastery here, just whether learning A
increases learning B.

## Test 3: Continual Learning

We want to see if the model forgets after a certain amount of time,
whether performance improved, and whether new knowledge interferes.
For example, we teach it A, B, C, then return to A to see if it
remembers it not perfectly, but enough to reproduce it on its own.

## Test 4: Discovery

We give the model minimal information and ask the system to infer
something not explicitly present for example, provide basic
measurements based on this minimal information.

## Test 5: Model Stability

We run the model for a longer time and verify if everything is
preserved. Are the fusion patterns stable?

## Test 6: Internal World Model

Does it build a coherent internal representation? Do the words it
outputs make sense together, not for us humans, but such that it
generally creates unknown rules we didn't enforce?

## Test 7: Adaptation Speed

Give environment A with certain constraints, then change one rule. See
how quickly the model adapts.

## Test 8: Meta-Learning

Basically, over time in unrelated domains, will the learning efficiency
increase?

## Test 9: Affective and Emotional Representations

In the code we use emotional and affective representations. We could
monitor them to see if something more gets encoded in them—not just
reward/loss, but whether deeper things emerge from the quite simple
emotional and affective structure.

## Constraints

Note that because the model isn't mainly a transformer model but more
of a state based model, we can't just observe these things in tests.
We will have to somehow observe them in the fused patterns. We already
do this to a certain extent in other modules. We need to remember that
this isn't going to be anywhere close to the human brain. We just want
to see if the emergence is deeper than in normal models. We shall have
to rerun trainings, probably because we test on interference—the
loaded model is then retrained. This is a very real possibility.
