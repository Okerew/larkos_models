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

## Test 10: Physics World Model

We run a live 2D physics simulation under normal gravity and feed the
model sequential observations as text—positions, velocities, masses.
The question is whether the fused representations actually track the
physical state: are states that are physically similar also similar in
the model's internal space? We measure this with Representational
Similarity Analysis between the physics state vectors and the fused
outputs. If the model builds a genuine world model, the alignment
should improve from early to late training and beat what a frozen
random projection could achieve.

## Test 11: Physics Rule Adaptation

We train the model on a normal-gravity physics world, checkpoint it,
then flip gravity upside down and continue training. The idea is to
see whether the internal world model reorganises to match the new
rules—measured both by loss recovery and by RSA re-alignment in fused
space. Passing means the model not only adapts behaviourally (loss
recovers) but also rebuilds its internal representation of physics to
match the new dynamics.

## Test 12: Affective Bond Interaction

We expose the model to four entities with distinct personalities: one
consistently positive, one consistently negative, one erratic, and one
neutral-steady. Each epoch the model interacts with one of them and we
measure the immediate emotional response (valence, love, hate,
surprise deltas) isolated from the training signal. The test checks
whether the affective system correctly tracks interaction quality—the
positive entity should trigger more love than hate, the negative one
more hate than love, and the erratic one should produce more surprise.
If the bonds are working, the per-entity valence deltas should line up
with the rewards each entity gives.

## Constraints

Note that because the model isn't mainly a transformer model but more
of a state based model, we can't just observe these things in tests.
We will have to somehow observe them in the fused patterns. We already
do this to a certain extent in other modules. We need to remember that
this isn't going to be anywhere close to the human brain. We just want
to see if the emergence is deeper than in normal models. We shall have
to rerun trainings, probably because we test on interference—the
loaded model is then retrained. This is a very real possibility.
