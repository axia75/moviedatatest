import gradio
import pandas as pd
import os
from videogit import videogif
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import os
from towhee import ops, pipe, register
from towhee.operator import PyOperator
from towhee import DataCollection
import gradio

show_num = 3
raw_video_path = '/work/dataset/test_1k_compress' # 1k test video path.
# test_csv_path = '/work/dataset/MSRVTT_JSFUSION_test.csv' # 1k video caption csv.

milvus_search_pipe = (
    pipe.input('sentence')
    .map('sentence', 'vec', ops.video_text_embedding.clip4clip(model_name='clip_vit_b32', modality='text', device='cpu'))
    .map('vec', 'rows', 
         ops.ann_search.milvus_client(
             host='127.0.0.1', port='19530', collection_name='text_video_retrieval_test', limit=show_num)
    )
    .map('rows', 'videos_path',
         lambda rows: (os.path.join(raw_video_path, 'video' + str(r[0]) + '.mp4') for r in rows))
    .output('videos_path')
)


def milvus_search_function(text):
    return milvus_search_pipe(text).to_list()[0][0]


# print(milvus_search_function('a girl wearing red top and black trouser is putting a sweater on a dog'))


interface = gradio.Interface(milvus_search_function, 
                             inputs=[gradio.Textbox()],
                             outputs=[gradio.Video(format='mp4') for _ in range(show_num)]
                            )

# interface.launch(inline=True, share=True)
interface.launch(server_name="0.0.0.0", server_port=8304,inline=True, share=False)
