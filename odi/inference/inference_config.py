import click
import json
from odi.model_util import odi_util as outil
import os

@click.command()
@click.option('--first_innings_emb',required=True,
              prompt='Use final embedding model for first innings True\False?',
              help='use embedding model if true',
              default=True,type=bool)
@click.option('--second_innings_emb',required=True,
              prompt='Use final embedding model for second innings True\False?',
              help='use embedding model if true',
              default=True,type=bool)
@click.option('--first_emb_model',required= True,
              prompt='Use player wise embedding or team wise embedding for first innings?',
              help='use team for team with batsman/advesarial for player wise',
              default='team')
@click.option('--second_emb_model',
                prompt='Use player wise embedding or team wise embedding for second innings?',
              help='use team for team with batsman/advesarial for player wise',
              default='team')
def create_inference_config(first_innings_emb,second_innings_emb,first_emb_model,second_emb_model):
    config = {
        'first_innings_emb':first_innings_emb,
        'second_innings_emb':second_innings_emb,
        'first_emb_model':first_emb_model,
        'second_emb_model':second_emb_model

    }

    json.dump(config,open(os.path.join(outil.DEV_DIR,'inference_config.json'),'w'),indent=2)

if __name__=="__main__":
    create_inference_config()
