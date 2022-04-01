

from tokenizers import decoders,models,normalizers,pre_tokenizers,processors,trainers,Tokenizer

tokenizer=Tokenizer(models.WordPiece(unk_token="[UNK]"))