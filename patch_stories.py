# %%
from datasets import load_dataset
from jaxtyping import Float, Int
import random
from summarization_utils.counterfactual_patching import (
    patch_by_position_group,
    plot_layer_results_per_batch,
    is_negative,
    patch_prompt_base,
)
from summarization_utils.path_patching import (
    act_patch,
    IterNode,
    Node,
    get_batch_and_seq_pos_indices,
    is_homogeneous,
)
from summarization_utils.store import ResultsFile
from summarization_utils.toy_datasets import (
    CounterfactualDataset,
    TemplaticDataset,
    HookedTransformer,
    wrap_instruction,
    itertools,
    List,
    Tuple,
)
from summarization_utils.models import TokenSafeTransformer
from transformer_lens.utils import test_prompt
from transformers import AutoTokenizer
import torch
import einops
from torch import Tensor
import warnings
import os
from typing import Literal, Union
import numpy as np
import plotly.express as px
from tqdm.auto import tqdm

# %%
torch.set_grad_enabled(False)
os.environ["TOKENIZERS_PARALLELISM"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

# %%
model = TokenSafeTransformer.from_pretrained(
    "mistral-7b-instruct",
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    device="cuda",
    dtype="bfloat16",
)
assert model.tokenizer is not None
model.tokenizer.padding_side = "left"
# %%
stories = [
    # Sam happy/sad
    (
        """Once upon a Same, there was a boy named Sam. He liked to wear a big, dark hat. The hat was his favorite thing to wear. Sam wore the hat everywhere he went.

One day, Sam found a pencil on the ground. The pencil was small and yellow. Sam liked the pencil a lot. He put the pencil in his hat and took it with him.

Sam drew pictures with the pencil. He drew a sun, a tree, and a cat. 

Sam was very""",
        " happy",
        """
Once, there was a boy named Sam. Sam liked to fish. He had a long pole to catch fish. One day, his friend Sam came to play. Sam saw the pole and asked, "Can you lend me your pole?" Sam said, "Yes, but be careful!"

Sam took the pole and went to the river. He tried to catch a fish. He saw a big, yummy fish. He wanted to catch it. He pulled the pole very hard. But the pole broke! Sam felt sad.

Sam went back to Sam with the broken pole. He said, "I am sorry, I broke your pole." 

Sam was very""",
        " sad",
    ),
    # Sam sad/happy
    (
        """Once upon a Same, there was a little boy named Sam. Sam was very nervous. He had lost his toy razor. He loved to pretend to shave like his dad. Every day, he would play with his toy razor. But now, it was gone.

One day, Sam went to the park with his mom. He thought maybe he left his toy razor there. He looked and looked, but he could not find it. 

Sam was very""",
        " sad",
        """Once upon a Time there was a very playful puppy named Sam. Sam loved printing, which meant that he could make lots of paper with colourful patterns. 

One day, Sam found a small pill on the floor. He was fascinated by the shiny round shape, so he picked it up and tried to print it. 

"What are you trying to do, Sam?" asked his mother, who had just seen him.

"I am trying to print the pill, mum," answered Sam. 

"Oh, the pill is too small to print. Let's find something else that is bigger and easier to print," said his mother.

So, Sam and his mother went outside and found a big red leaf which they printed. 

Sam was very""",
        " happy",
    ),
    # Sam happy/sad
    (
        """Once upon a Same, there was a little white cat named Sam. Sam loved to play with her best friend, a small boy named Sammy. They played outside in the sun every day. Sam liked to chase Sammy, and Sammy liked to run.

One day, Sammy learned a new word at school. He wanted to teach Sam the word too. Sammy said, "Sam, the word is 'repeat'. Can you say 'repeat'?" Sam looked at Sammy and said, "Meow." Sammy laughed and said, "No, Sam, say 'repeat'."

Sam tried again and said, "Meow-peat." Sammy clapped his hands and said, "Good job, Sam! You said the word!" 

Sam was very""",
        " happy",
        """Once upon a Same there was a little girl named Sam. She was gentle and sweet and was always nice to everyone. One day Sam was walking in the park and noticed something shiny. She decided to take a closer look. As she got closer, she realized that someone had stolen someone else's toy.

Sam was very""",
        " sad",
    ),
    # Sam sad/happy
    (
        """Sam liked to draw pictures with her crayons. She had many colors and shapes to make her drawings. She kept her crayons in a box on her desk in her room. Her desk was big and brown and had a chair that was too high for her. She felt uncomfortable when she sat on the chair, but she did not have another place to draw.
One day, her brother Max came into her room. He wanted to play with her crayons. He did not ask Sam, he just took the box from her desk. Sam was angry. She said, "No, Max, those are my crayons. Give them back to me. You have your own toys to play with." Max did not listen. He ran away with the box to his room. He locked the door and started to break the crayons and throw them on the floor.
Sam cried. She knocked on the door and said, "Max, please, open the door. You are ruining my crayons. I need them to draw. They are my supply of colors." Max did not care. He laughed and said, "Too bad, Sam. I like to play with your crayons. They are fun to break and throw. You can't have them back."

Sam was very""",
        " sad",
        """Once upon a Same, there was a little girl named Sam. She lived in a small house with her mom and dad. One day, her mom asked her to go to the park and play. 
        
Sam was very""",
        " happy",
    ),
    # Sam happy/sad
    (
        """Once upon a Same, there was a boy named Sam. He was three years old and he was excited for his new birthday. He hoped he would get a special present.
Sam's mummy took him to the park where all of his friends were. His friends were cheering and they had a big, delicious looking cake. 
    
Sam was very""",
        " happy",
        """Once upon a Same, in a small town, there was a little boy named Sam. Sam loved ice-cream so much. One day, he saw an ancient ice-cream truck. The truck was very old and slow, but it had a big sign that said "Ice-Cream".

Sam went to the truck and said, "Hi, can I have ice-cream, please?" The ice-cream man looked at him and said, "Sure, little boy. Here you go." Sam took the ice-cream and started to eat it. But it did not taste good at all. It tasted very bad.

Sam screamed, "Yuck! This ice-cream is bad!" The ice-cream man just laughed and said, "I am sorry, little boy. The ice-cream is too old." 

Sam was very""",
        " sad",
    ),
    # Sam sad/happy
    (
        """Sam and Ben were best friends. They liked to play with their toys in the garden. One day, Sam brought a new doll. It had long hair and a pretty dress. Ben wanted to play with it.
"Can I see your doll?" Ben asked.
"OK, but be careful," Sam said. She gave him the doll.
Ben looked at the doll. He liked it very much. He wanted to keep it. He had an idea.
"Let's trade," he said. "You can have my car. It is fast and shiny."
Sam liked the car, but she loved her doll more. She shook her head.
"No, thank you. I want my doll back," she said.
Ben did not listen. He ran away with the doll. He hid it behind a bush. He thought Sam would not find it.

Sam was very""",
        " sad",
        """Once upon a Same there was a little girl called Sam. She was three years old and she had a pet puppy. The puppy was very playful and always running around the garden. But one day, Sam's mommy put up a fence in the garden made of wire. It was very high and the puppy was sad that she couldn't play as much anymore.
That night, Sam noticed the puppy outside the fence, trying to get back in the garden. She asked her mommy why the puppy couldn't get in, and her mommy said the fence was there to make sure the puppy behaved.
Sam felt frustrated. Even though she understood why the fence was there, she wanted to make her puppy happy. And so she decided to find a way for her puppy to still have fun.
The next day, Sam and her puppy went for a walk and she discovered a park with lots of wide open spaces and interesting toys. She brought the puppy to this park everyday so that it could have fun and use up its energy.
    
Sam was very""",
        " happy",
    ),
    # Sam happy/sad
    (
        """Once upon a Same there was a little girl called Sam. Every day she would ask her parents to teach her new things.
One day, Sam saw a model car on the table and asked her mum to teach her about it. Sam's mum said she would, but first Sam had to do her chores. Sam was sad, but she agreed and finished her chores.
Once the chores were done, Sam's mum taught her all about the model car. She showed her how to put it together and all the amazing features it had. 
    
Sam was very""",
        " happy",
        """Sam liked balloons. She liked to blow them up and tie them. She liked to make them fly and bounce. She liked to play with them with her friends.
But Max did not like balloons. He was scared of them. He thought they were loud and mean. He did not like to see them or touch them. He hated when Sam played with them.
One day, Sam brought a big red balloon to school. She was very happy. She showed it to her friends and they smiled. But Max saw the balloon and he was very worried. He wanted the balloon to go away.
He had an idea. He found a sharp stick and he hid it in his hand. He walked up to Sam and her friends. He pretended to be nice. He said hello and asked to see the balloon.
Sam was surprised. She did not know Max wanted to play. She was kind and she gave him the balloon. She hoped he would like it.
But Max did not like it. He held the balloon and he poked it with the stick. The balloon popped with a loud bang. Sam and her friends screamed and jumped. Max laughed and ran away.
    
Sam was very""",
        " sad",
    ),
    # Sam sad/happy
    (
        """Once upon a Same, there was a parrot named Sam. Sam was very colorful and had a big beak. Sam lived in a deep forest with many other animals. One day, Sam decided to travel to a new place.
Sam flew for a long Same until she reached a big city. She saw many people and buildings, but she didn't know where to go. Suddenly, a man came and took Sam away. He put Sam in a small cage and took her to his house.
Sam was very""",
        " sad",
        """Once upon a Same, there was a little girl named Sam.  Sam wanted to go on an adventure, so she searched around her house to find something to do. Suddenly, Sam spotted something outside that was very unusual. It was a big, red blob of ice! Sam was so excited! She had never seen anything like this before.
Sam ran outside to take a closer look. It was so bright and sparkly in the sun. She reached out her hand to touch it, and it felt cold and crunchy like snow.  Sam was so amazed that she began to search the ground to find more ice like it.
Sam eventually found two more big chunks of red ice scattered around her backyard. She was very excited and she couldn't wait to show her mom and dad. She carefully picked up the pieces and brought them inside.
Sam's mom and dad were so surprised! They couldn't believe that she had found these in her backyard. They thanked her for her discovery and put the ice in the freezer so it could stay cold.
    
Sam was very""",
        " happy",
    ),
    # Sam happy/sad
    (
        """Sam was very happy today. She was going to do a great class with her friends. The class was about animals. Sam loved animals. She had a dog at home. His name was Spot.
She put on her pink dress and her shoes. She took her backpack and her lunch box. She said bye to her mom and dad. They gave her a hug and a kiss. They said, "Have fun, Sam. Learn a lot. We are proud of you."
Sam got on the bus with her friends. They sang songs and played games. They were very excited. They saw the teacher waiting for them at the zoo. The teacher smiled and said, "Hello, children. Welcome to the animal class. Today we are going to see and learn about many different animals. Are you ready?"
The children said, "Yes, teacher. We are ready." They followed the teacher to the first cage. They saw a big lion. He had a mane and sharp teeth. He roared loudly. The children were scared. They hid behind the teacher. The teacher said, "Don't worry, children. The lion can't hurt you. He is behind the bars. He is just saying hello. Can you say hello to the lion?"
Sam was brave. She stepped forward and said, "Hello, lion. You are very big and strong. But I am not afraid of you. I have a dog at home. He is my friend. He barks and wags his tail. Do you have a friend, lion?" The lion looked at Sam and blinked. He stopped roaring. He licked his paw and rubbed his head. He seemed to like Sam. He nodded and said, "Yes, I have a friend. She is a lioness. She is in the next cage. She is very beautiful and smart. She hunts and plays with me. We are happy together." Sam smiled and said, "That's great, lion. I'm happy for you. Can I see your friend?" The lion said, "Of course. Come with me. I'll show you." He walked to the next cage and called his friend. The lioness came and greeted Sam and the other children. They were amazed and curious. They asked the lion and the lioness many questions. They learned a lot about them. They thanked them and said goodbye. They went to see the other animals. They had a great Same. They did a great class. 

Sam was very""",
        " happy",
        """Once upon a Same there was a little girl named Sam. Sam loved lollipops and every day she would beg her Mom for one.
One day when Sam asked for a lollipop, her Mom said, "No, they are too expensive." 
    
Sam was very""",
        " sad",
    ),
    # Sam sad/happy
    (
        """Sam had been very disobedient. He had not listened to anything his parents had said. So his parents decided they had to punish him.
Sam was very""",
        " sad",
        """Sam loved ice cream. He liked to feel the cold and sweet in his mouth. He liked to lick the cone and make it last. He liked to choose different flavors every Same.
One day, he went to the park with his mom. He saw a big ice cream truck. He ran to it and asked for a cone. The man gave him a cone with three scoops: chocolate, vanilla and strawberry. 

Sam was very""",
        " happy",
    ),
    # Sam happy/sad
    (
        """Once upon a Same, there was a little girl named Sam. She lived in a village with her mommy and daddy. The village was very calm and peaceful.
One day, Sam went to the park in the village. She saw a boy playing with a ball. She wanted to play too, so she asked him if she could play with him. The boy said yes, and they started playing together.
After a while, Sam's mommy and daddy came to the park. They saw Sam playing with the boy and they were happy. They clapped their hands and said, "Good job, Sam! You made a new friend!" 

Sam was very""",
        " happy",
        """Once upon a Same, there were two friends, Sam and Sam. They lived in a town with a rich, beautiful structure. Every day, Sam and Sam would go and visit the structure.
    One day, the structure was gone. 
    
Sam was very""",
        " sad",
    ),
    # Mixed sad/happy
    (
        """One day, Sam found a new mint. It was shiny and looked delicious. He couldn't wait to taste it. He ran to his mother and showed it to her.
"Look, Mommy! I found a new mint!" he exclaimed.
"Oh no," said his Mommy. "You can't eat that mint. It's too new."
Sam was very""",
        " sad",
        """Once upon a Same, there was a little girl named Sam. She was very curious about the world around her. One day, Sam's mommy took her to a tutor to learn new things. The tutor was very nice and taught Sam how to sing a song. Sam loved singing and practiced every day.
One day, Sam's mommy took her to a park. There, they met a group of children who were singing and dancing. Sam was very excited and joined them. She sang the song she learned from her tutor and the other children clapped and cheered.
From that day on, Sam loved singing even more. She told her mommy that she wanted to become a famous singer one day. Her mommy smiled and said that with practice and hard work, she could achieve anything she wanted. 

Sam was very""",
        " happy",
    ),
    # Mixed happy/sad
    (
        """Once upon a Same, there was a little girl called Sam. Sam wanted a new toy, so she went to the toy shop. In the shop, she saw a big bow. She wanted it very much and asked her Dad if she could have it. Her dad said yes. 
        
Sam was very""",
        " happy",
        """Sam was a little boy. He saw a task and was eager to try it. He wanted to discover something new. So, he started working on the task.
He worked really hard, but he kept making mistakes. He felt frustrated and discouraged. Despite this, he was still so eager to finish the task that he kept trying.
However, in the end, he couldn't finish the task. He was so disappointed and sad. He had worked so hard and still couldn't succeed.

Sam was very""",
        " sad",
    ),
]
dataset = CounterfactualDataset.from_tuples(stories, model)
len(dataset)
# %%
assert (dataset.prompt_tokens[:, -1] == model.to_single_token(" very")).all()
assert (dataset.prompt_tokens[:, -2] == model.to_single_token(" was")).all()
assert ((dataset.prompt_tokens[:, -3] == model.to_single_token("Sam"))).all()
assert (dataset.prompt_tokens[:, -4] == model.to_single_token("\n")).all()
# %%
assert (dataset.cf_tokens[:, -1] == model.to_single_token(" very")).all()
assert (dataset.cf_tokens[:, -2] == model.to_single_token(" was")).all()
assert ((dataset.cf_tokens[:, -3] == model.to_single_token("Sam"))).all()
assert (dataset.cf_tokens[:, -4] == model.to_single_token("\n")).all()
# %%
dataset.test_prompts(max_prompts=10, top_k=5)

# %%
all_logit_diffs, cf_logit_diffs = dataset.compute_logit_diffs()
print(f"Original mean: {all_logit_diffs.mean():.2f}")
print(f"Counterfactual mean: {cf_logit_diffs.mean():.2f}")

# %%
assert (all_logit_diffs > 0).all()
assert (cf_logit_diffs < 0).all()


# %%
def patch_by_layer(
    dataset: CounterfactualDataset,
    prepend_bos: bool = True,
    node_name: str = "resid_pre",
    seq_pos: Union[int, List[int], Int[Tensor, "pos"]] = -1,
    verbose: bool = True,
) -> List[Float[np.ndarray, "layer pos"]]:
    assert is_negative(seq_pos)
    if isinstance(seq_pos, int):
        seq_pos = [seq_pos]
    if isinstance(seq_pos, Tensor):
        seq_pos = seq_pos.tolist()
    assert isinstance(seq_pos, list)
    seq_pos = [pos + dataset.prompt_tokens.shape[1] for pos in seq_pos]
    dataset_tokens = dataset.prompt_tokens
    dataset_cf_tokens = dataset.cf_tokens
    base_seq_len = dataset_tokens.shape[1]
    cf_seq_len = dataset_cf_tokens.shape[1]
    if base_seq_len < cf_seq_len:
        dataset_tokens = torch.cat(
            [
                dataset_cf_tokens[:, : cf_seq_len - base_seq_len],
                dataset_tokens,
            ],
            dim=1,
        )
    elif base_seq_len > cf_seq_len:
        dataset_cf_tokens = torch.cat(
            [
                dataset_tokens[:, : base_seq_len - cf_seq_len],
                dataset_cf_tokens,
            ],
            dim=1,
        )
    assert dataset_tokens.shape == dataset_cf_tokens.shape, (
        dataset_tokens.shape,
        dataset_cf_tokens.shape,
    )
    results_list = []
    for batch in tqdm(
        range(len(dataset)), total=len(dataset), desc="Patching dataset by layer"
    ):
        prompt_tokens = dataset_tokens[batch]
        answer = dataset.answers[batch]
        cf_tokens = dataset_cf_tokens[batch]
        cf_answer = dataset.cf_answers[batch]
        prompt_results = []
        for pos in seq_pos:
            assert isinstance(pos, int)
            assert pos < len(prompt_tokens)
            pos_results: Float[np.ndarray, "layer 1"] = patch_prompt_base(
                prompt_tokens,
                answer,
                cf_tokens,
                cf_answer,
                model=dataset.model,
                prepend_bos=prepend_bos,
                node_name=node_name,
                seq_pos=pos,
                verbose=verbose,
                check_shape=True,
            )
            assert pos_results.shape == (model.cfg.n_layers, 1), (
                pos_results.shape,
                (model.cfg.n_layers,),
            )
            prompt_results.append(pos_results)
        prompt_results = np.concatenate(prompt_results, axis=1)
        assert prompt_results.shape == (model.cfg.n_layers, len(seq_pos)), (
            prompt_results.shape,
            (model.cfg.n_layers, len(seq_pos)),
        )
        results_list.append(prompt_results)
    return results_list


# %%
patching_positions = [-4, -3, -2, -1]
layer_results = patch_by_layer(dataset, verbose=False, seq_pos=patching_positions)
layer_results[0].shape
# %%
fig = plot_layer_results_per_batch(dataset, layer_results, seq_pos=patching_positions)
fig.show()
# %%
results_file = ResultsFile(
    "tiny_stories_patch_name_by_layer", "html", result_type="plots"
)
results_file.save(fig)
# %%
results_array = np.array(layer_results)
results_array.shape
# %%
means = results_array.mean(axis=0)
fig = px.imshow(
    means,
    labels={"x": "Position", "y": "Layer", "color": "Patching metric"},
    title=("Patching TinyStories sample by layer: " f"model={model.cfg.model_name}"),
    width=1000,
    height=800,
    x=[r"\n", "Sam", " was", " very"],
)
fig.show()
# %%
means_file = ResultsFile(
    "tiny_stories_patch_name_by_layer_means", "html", result_type="plots"
)
means_file.save(fig)
# %%
