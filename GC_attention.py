from lstm import *
from termcolor import colored
from collections import OrderedDict
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertAdam
class GC_attention(LSTM_Backen):
	def __init__(self,bucket,algorithm):
		super(GC_attention,self).__init__()
		self.optimizer='adam'
		self.loss_func='xent_hinge'
		
		training=True
		if training:
			self.answer_piece=1
		else:
			self.answer_piece=8

		self.char_channels=200
		self.max_query=40
		self.max_answer=200
		# self.max_bucket=int(1e+10)
		self.max_bucket=100
		# <RL>
		self.max_sent_num=bucket
		# </RL>

		plt.title('lstm')
		# print self._get_name()
		# share one representation for 2 tasks
		self.model_path=self.model_dir+'/model/qa'
		print ('Load pre-trained model tokenizer (vocabulary)')
		self.bt_tokenizer = BertTokenizer.from_pretrained('/home/zack/bert-base-uncased')
		print ('Load pre-trained model (weights)')
		
		self.bt_model = BertModel.from_pretrained('/home/zack/bert-base-uncased')
		# device=torch.device('cuda:0')
		# self.bt_model.to(device)
		self.bt_model=self.bt_model.eval()

		# for save test file name when only use test not train
		if algorithm=='RAF+Base+Static+GC':
			self.add_model_vars()
			self.add_model_class()
	
	def build(self,feed_dict):
		"""
		this function only invoke or initialize the functions without parameter, 
		or use __init__ hyper parameter to fill the parameter
		"""
		self.cache=OrderedDict()
		self.feed_dict=feed_dict
		# self.lstm_mix()
		self.fcn()
		if self.training:
			self.loss_layer()

	def forward(self,feed_dict):
		self.build(feed_dict)

	def add_model_vars(self):
		# embeddings_word=torch.cuda.FloatTensor(self.word_embed)
		# self.embeddings_word=torch.nn.Embedding.from_pretrained(embeddings_word,
		# 	freeze=True).cuda()

		# self.embeddings_build=torch.nn.Embedding(self.build_size,
		# 	self.word_dim).cuda()
		# self.init_embedding(self.embeddings_build.weight)

		# self.embeddings_char=torch.nn.Embedding(self.char_types,
		# 	self.char_embed_dim).cuda()
		# self.init_embedding(self.embeddings_char.weight)

		# self.overlap_embedding=torch.nn.Embedding(2,
		# 	self.overlap_dim).cuda()
		# self.init_embedding(self.overlap_embedding.weight)

		# =======attend======
		dim=self.lstm_hidden_dim*2
		# self.w1=torch.nn.Parameter(torch.cuda.FloatTensor(
		# 	dim,dim),
		# 	requires_grad=True)
		# self.init_weight(self.w1)
		# self.w2=torch.nn.Parameter(torch.cuda.FloatTensor(
		# 	dim,dim).cuda(),
		# 	requires_grad=True)
		# self.init_weight(self.w2)

		self.doc_observer=torch.nn.Parameter(torch.cuda.FloatTensor(
			1,200).cuda(),
			requires_grad=True)
		self.init_weight(self.doc_observer)

		# self.query_observer=torch.nn.Parameter(torch.cuda.FloatTensor(
		# 	1,dim).cuda(),
		# 	requires_grad=True)
		# self.init_weight(self.query_observer)

		# self.pair_sum_vec=torch.nn.Parameter(torch.cuda.FloatTensor(
		# 	1,dim*2).cuda(),
		# 	requires_grad=True)
		# self.init_weight(self.pair_sum_vec)

	def add_model_class(self):
		# =====bn=====
		bert_dim=768
		
		self.linear = torch.nn.Linear(bert_dim+200 , 1).cuda()
		self.init_linear(self.linear)

		self.mlp_bert = torch.nn.Linear(bert_dim , 200).cuda()
		self.init_linear(self.mlp_bert)

		# self.mlp_bert2 = torch.nn.Linear(bert_dim , 200).cuda()
		# self.init_linear(self.mlp_bert2)

		self.linear_filter = torch.nn.Linear(bert_dim , 1).cuda()
		self.init_linear(self.linear_filter)

		# self.lstm_bert=torch.nn.LSTM(input_size= 768,
		# 	hidden_size=100,bidirectional=True,batch_first=True).cuda()
		# self.init_lstm(self.lstm_bert)

		self.dropout = torch.nn.Dropout(0.5)
		
		decay_learning_rate=self.lr

		params_dict=dict(self.named_parameters())
		parameter_list=[]
		optim_dict=OrderedDict()
		for k,v in params_dict.items():
			if re.search(r'embeddings_word',k):
				# print k
				pass
			else:
				# print (colored('I init varibales again','red'))
				# print (v.size(),k)
				# if re.search(r'bt_model',k):
				# 	pass
				# elif re.search(r'bt_tokenizer',k):
				# 	pass
				# else:
				# 	# if len(v.size())>1:
				# 	# 	torch.nn.init.uniform(v)
				# 	# else:
				# 	# 	torch.nn.init.uniform(v)
				# 	# 	print (v.size(),k)
				# 	pass
				parameter_list.append(v)
				optim_dict[k]=v

				# print ('{}->{}'.format(k,v.requires_grad))
		self.parameter_list=parameter_list
		self.optim_dict=optim_dict

		# self.mseloss=torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
		self.logloss=torch.nn.BCELoss()
		# self.xentloss=torch.nn.CrossEntropyLoss()
		# self.nllloss=torch.nn.NLLLoss()
		
		self.optimizer_ = BertAdam(parameter_list,
			lr=decay_learning_rate,
			warmup=0.001,
			t_total=10000000
			)

		# print '{}?{}'.format(self.optimizer_,self._get_name())
	
	def batch_bert(self,data,batch_size=12):
		tmp=[]
		i=0
		while i < len(data['index']):
			batch={'pairs_token_memory':[],'paris_segs_memory':[],'index':[]}

			pairs_token_memory=data['pairs_token_memory'][i:i+batch_size]
			paris_segs_memory=data['paris_segs_memory'][i:i+batch_size]
			act_query_length=data['act_query_length']
			act_answer_length=data['act_answer_length'][i:i+batch_size]

			batch_query_length=np.amax(act_query_length)
			batch_answer_length=np.amax(act_answer_length)

			batch['index']=data['index'][i:i+batch_size]
			# act_query_length contains=tokens+['[CLS]']+['[SEP]']
			# act_answer_length=tokes+['[CLS]'] so the sum will contains one more pad token
			lens=batch_query_length+batch_answer_length-1
			batch['pairs_token_memory']=np.array(pairs_token_memory)[:,:lens]
			batch['paris_segs_memory']=np.array(paris_segs_memory)[:,:lens]
			tmp.append(batch)

			i=i+batch_size
		return tmp

	def update_memory(self,a,positive_indics,
		topk_ids_pos,topk_value_pos,
		topk_ids_neg,topk_value_neg,
		berts_pos,berts_neg,
		i,flg,r):

		if self.training==False:
			positive_indics=[]

		if i in positive_indics:
			topk_ids_pos.append(i)
			topk_value_pos.append(flg)
			berts_pos.append(r)
			assert(len(berts_pos)==len(topk_ids_pos)==len(topk_value_pos))

		# if more than topk start filter
		else:
			
			if len(topk_ids_neg)==0:
				# this place we use one sentence to describe in paper
				# topk_value_neg=float('-inf') init all three variables
				topk_value_neg.append(flg)
				topk_ids_neg.append(i)
				berts_neg.append(r)
				return topk_ids_pos,topk_value_pos,topk_ids_neg,topk_value_neg,berts_pos,berts_neg

			# for i_n in range(len(topk_value_neg)):
			i_n=0
			print('flg',flg)
			while flg > topk_value_neg[i_n]:
				i_n+=1
				if i_n >= len(topk_value_neg):
					break
			# < and = put it down
			# this place we use a word description in paper
			topk_value_neg.insert(i_n,flg)
			topk_ids_neg.insert(i_n,i)
			berts_neg.insert(i_n,r)
			
			assert(len(berts_neg)==len(topk_ids_neg)==len(topk_value_neg))

		neg_volume=self.max_sent_num-len(topk_ids_pos)
		if len(topk_ids_neg)+len(topk_ids_pos)>self.max_sent_num:
			bar=-neg_volume
			# this assign operation can't return automatically
			topk_value_neg=topk_value_neg[bar:]
			topk_ids_neg=topk_ids_neg[bar:]
			berts_neg=berts_neg[bar:]

		print ('|pos_id|',topk_ids_pos,positive_indics,'|topk_neg_ids|',topk_ids_neg,len(topk_value_neg),
			'|topk_bert_vectors|',len(berts_neg),'|topk_neg_values|',topk_value_neg)

		return topk_ids_pos,topk_value_pos,topk_ids_neg,topk_value_neg,berts_pos,berts_neg
				
	def lstm_batches(self,batches,positive_indics):
		berts_pos=[]
		berts_neg=[]
		topk_ids_pos=[]
		topk_value_pos=[]
		topk_ids_neg=[]
		topk_value_neg=[]

		for batch in batches:
			
			pair_tokens=batch['pairs_token_memory']
			pair_segments=batch['paris_segs_memory']
			print (batch)
			index=batch['index']
			print('241')
			encoded_layers=self.bert_embed_layer(pair_tokens,pair_segments)
			# (batch,1,768)
			# r_batch=encoded_layers[-1][:,0]
			# encoded_layers=encoded_layers[-1][:,0]
			flg=self.linear_filter(encoded_layers)
			
			# if two value both large, its prob is just 0.5 so softmax is necessary
			# if using sigmoid it's a regression problem not classification, regression is bad result
			# flg=torch.nn.functional.softmax(flg,-1)

			for i,ind in enumerate(index):
				r=torch.unsqueeze(encoded_layers[i],dim=0)
				# print('r (1,768)',r.size())
				f=flg[i]

				topk_ids_pos,topk_value_pos,topk_ids_neg,topk_value_neg,berts_pos,berts_neg= \
				self.update_memory('placeholder',positive_indics,
					topk_ids_pos,topk_value_pos,
					topk_ids_neg,topk_value_neg,
					berts_pos,berts_neg,
					ind,f,r)
				
		topk_ids=[]
		topk_value=[]
		bert=[]

		topk_ids=topk_ids+topk_ids_pos+topk_ids_neg
		topk_value=topk_value+topk_value_pos+topk_value_neg
		bert=bert+berts_pos+berts_neg

		print('topk_ids',len(topk_ids),len(topk_ids_pos),len(topk_ids_neg),topk_ids_neg)
		# if all positive sentences are larger than bucket size, it will error
		assert(len(topk_ids)<=self.max_sent_num)

		topk_ids_tsr=torch.cuda.LongTensor(np.array(topk_ids))
		# tensor id is sorted but not tensor values
		sorted_ids,sorted_index=topk_ids_tsr.sort(descending=False)
		# print('topk_value',topk_value)
		topk_value_tsr=torch.cat(topk_value,dim=0)
		topk_value_tsr=topk_value_tsr[sorted_index]

		bert_tsr=torch.cat(bert,dim=0)
		bert_tsr=bert_tsr[sorted_index]
		assert(sorted_ids.size(0)==topk_value_tsr.size(0)==bert_tsr.size(0))
		# print('bert_tsr (batch,768)',bert_tsr.size())

		return sorted_ids,topk_value_tsr,bert_tsr

	def fine_layer(self,memory,topk_ids,topk_berts):
		local_context_tokens=[]
		segments_ids_tokens=[]
		
		bucket_memory=[(memory['answer_token_memory'][i],
			memory['answer_segs_memory'][i]
			# you made a mistake here answer_segs_memory is 1 ,but we need 0
			# [0]*len(memory['answer_token_memory'][i])
			,memory['act_answer_length'][i]) for i in topk_ids]

		assert (len(bucket_memory)<=self.max_sent_num)

		bucket_tsr=[]
		i=0
		while i < len(bucket_memory):

			tokens=[e[0] for e in bucket_memory[i:i+self.answer_piece]]
			segments=[e[1] for e in bucket_memory[i:i+self.answer_piece]]
			answer_length=[e[2] for e in bucket_memory[i:i+self.answer_piece]]
			# the last one can't be empty
			assert(len(tokens[-1])>0)
			assert(len(segments[-1])>0)
			batch_answer_length=np.amax(answer_length)
			tokens=np.array(tokens)[:,:batch_answer_length]
			segments=np.array(segments)[:,:batch_answer_length]
			assert(tokens.shape[-1]>0)
			assert(segments.shape[-1]>0)
			print ('segments.shape',tokens,segments)

			# print ('301 line')
			# print (tokens,segments,answer_length)
			encoded_layers=self.bert_embed_layer(tokens,segments)
			# eliminate the other tensors to save memory
			# encoded_layers=encoded_layers[-1][:,0]
			i=i+self.answer_piece
			bucket_tsr.append(encoded_layers)

		answers_tsr=torch.cat(bucket_tsr,0)
		assert (topk_berts.size(0)==answers_tsr.size(0))

		answers_lcontext=torch.unsqueeze(answers_tsr,0)
		# topk_berts=torch.unsqueeze(topk_berts,dim=0)
		# print ('[answers total]',answers_lcontext.size())
		# o,(c1,c2)=self.lstm_bert(answers_lcontext)
		# print ('----just using mlp----')
		o=self.mlp_bert(answers_lcontext)
		print('o.size',o.size())
		# o=torch.relu(o)
		# o=torch.tanh(o)
		# o=self.gelu(o)
		# o=self.gelu(o)
		# print('topk_berts.size()',topk_berts.size())
		# assert(topk_berts.size(0)==o.size(0))
		# assert(topk_berts.size(1)==o.size(1))
		# o/answers
		# o: no query incorporate
		# answers: with query incorporate
		# alpha=torch.nn.functional.softmax(self.linear_self(o),0)
		
		fin=self.gcattention(topk_berts,o)
		return fin


	def lstm_mix(self):
		print('===== in the net =====',__file__)
		q=self.feed_dict['question_text']
		a=self.feed_dict['answer_text']
		print('answer number',len(a))

		positive_indics=[]

		if self.training:
			l=torch.squeeze(torch.cuda.FloatTensor(self.feed_dict['label']),dim=0)
			l=l.view([-1])
			positive_indics=(l>0.5).nonzero()
			# positive_indics=positive_indics.view([-1])
			print ('positive_indics',positive_indics,l)

		memory={'pairs_token_memory':[],'paris_segs_memory':[],'index':[],
		'answer_token_memory':[],'answer_segs_memory':[],
		'act_query_length':[],'act_answer_length':[]}

		# query
		q_indexed_tokens,q_segments_ids,act_query_length=self.bert_ids(q[0],0,query='query',pad_len=0)
		q_indexed_tokens=q_indexed_tokens[:self.max_query]
		q_segments_ids=q_segments_ids[:self.max_query]
		# print ('query',q_indexed_tokens)
		memory['act_query_length'].append(act_query_length)

		# paris_segs_memory=[]
		for i,a_ in enumerate(a):
			# print (q,a_)
			pair_tokens=[]
			pair_segments=[]
			# print ('query segments_ids length',len(segments_ids))
			pair_tokens=pair_tokens+q_indexed_tokens
			pair_segments=pair_segments+q_segments_ids
			# print ('answer')
			indexed_tokens,segments_ids,act_answer_length=self.bert_ids(a_,1,pad_len=self.max_query+self.max_answer)
			# print (indexed_tokens)
			# it might abandon a sentence with strange token
			# print ('indexed_tokens',indexed_tokens,'seg',segments_ids)
			if act_answer_length==0:
				print ('pad zero length',a_)
				indexed_tokens=[0]
				segments_ids=[1]
				act_answer_length=1
				# sys.exit(0)
				# continue
			# print ('answer segments_ids length',len(segments_ids))
			pair_tokens=pair_tokens+indexed_tokens
			pair_tokens=pair_tokens[:self.max_query+self.max_answer]
			pair_segments=pair_segments+segments_ids
			pair_segments=pair_segments[:self.max_query+self.max_answer]

			assert (len(pair_tokens)==len(pair_segments))
			# print (pair_tokens)
			memory['pairs_token_memory'].append(pair_tokens)
			memory['paris_segs_memory'].append(pair_segments)
			memory['index'].append(i)
			# ====create answer ====
			# 101:['CLS']
			# print("pad ['CLS']=101 ['SEP']=102")
			answer=[101]+indexed_tokens
			memory['act_answer_length'].append(act_answer_length+1)
			# memory['act_answer_length'].append(act_answer_length)

			answer_token=answer[:self.max_answer]
			answer_segs=[0]*len(answer_token)
			memory['answer_token_memory'].append(answer_token)
			memory['answer_segs_memory'].append(answer_segs)

		assert(len(memory['act_answer_length'])==len(memory['answer_token_memory']))
		if self.training:
			assert(len(memory['index'])==l.size(0))

		batches=self.batch_bert(memory,batch_size=self.answer_piece)
		topk_ids,topk_values,topk_berts=self.lstm_batches(batches,positive_indics)
		print('sorted',topk_ids,topk_values)
		# self.topk_value=torch.unsqueeze(topk_values,dim=-1)
		# self.topk_value=torch.nn.functional.softmax(topk_values,0)
		self.topk_value=torch.sigmoid(topk_values)
		self.inds=topk_ids

		one=False
		if one==False:
		
			fin=self.fine_layer(memory,topk_ids,topk_berts)		
			assert(fin.size(0)==topk_values.size(0))

			fin_=self.linear(fin)
			# y2_=torch.nn.functional.softmax(fin_,-1)
			# y2=torch.unsqueeze(y2_[:,1],-1)
			# y2=torch.nn.functional.softmax(y2,0)

			y2=torch.nn.functional.softmax(fin_,0)
			# y2=torch.sigmoid(fin_)

		else:
			y2=torch.unsqueeze(self.topk_value,-1)

		print ('y2.size',y2)
		# y2=torch.sigmoid(fin_)
		# print(y2.view([-1]))

		self.pro_tmp=y2
		if len(a)>self.max_sent_num:
			r=torch.zeros([len(a),1]).cuda()
			print ('r[self.inds].size()',r[self.inds].size())
			r[self.inds]=y2
			self.pro=r
		else:
			self.pro=y2
		print ('new pro',self.pro)
		assert(self.pro.size(0)==len(a))


	def fcn(self):
		self.lstm_mix()
		

	def gcattention(self,answers,o):
		doc_observer=torch.unsqueeze(self.doc_observer,0)
		global_context,_,_,alpha=self.attend(doc_observer,o)
		print(o.size(),answers.size())
		# global_context,_,_,alpha=self.attend(doc_observer,torch.unsqueeze(self.mlp_bert2(answers),0))
		# print ('[alpha]',alpha.size())
		print ('----finish one alpha----')
		# o/answers
		# o: no query incorporate
		# global_context=torch.sum(alpha*o,0)

		doc=torch.squeeze(global_context,0)

		ones=torch.ones([answers.size(0),1]).cuda()
		doc=ones*doc

		# this global attention is unique since it doesn't have additional parameters
		# answers_concat=torch.cat([torch.squeeze(o,0),doc],-1)
		answers_concat=torch.cat([answers,doc],-1)

		global_att=torch.matmul(answers_concat,torch.transpose(answers_concat,0,1))

		global_att=torch.nn.functional.softmax(global_att,-1)

		global_att=torch.unsqueeze(global_att,-1)

		# incorporate=global_att*torch.unsqueeze(answers_concat,0)

		# fin=torch.sum(incorporate,1)

		incorporate=global_att*o
		fin=torch.sum(incorporate,1)

		fin=torch.cat([answers,fin],-1)
		return fin

	def bert_ids(self,sent,id_,query='None',pad_len=0):
		act_length=0
		if query=='query':
			tokens,act_length=self.bert_vector(sent,s_token=['[CLS]'],e_token=['[SEP]'],pad_len=pad_len)
		if query=='single':
			tokens,act_length=self.bert_vector(sent,s_token=['[CLS]'],e_token=['[SEP]'],pad_len=pad_len)
		if query=='None':
			tokens,act_length=self.bert_vector(sent,e_token=['[SEP]'],pad_len=pad_len)

		# Define sentence A and B indices associated to 1st and 2nd sentences (see paper)
		segments_ids = [id_]*len(tokens)
		indexed_tokens= tokens
		return indexed_tokens,segments_ids,act_length


	def bert_vector(self,sent,s_token=None,e_token=None,pad_len=0):
		# print (sent)
		tokenized_text = self.bt_tokenizer.tokenize(sent)

		tokens=tokenized_text
		if s_token is not None:
			tokens=s_token+tokens
		if e_token is not None:
			tokens=tokens+e_token
		act_length=len(tokens)

		if pad_len>0:
			dif=pad_len-len(tokens)
			if dif>0:
				pad=['[PAD]']*dif
				tokens=tokens+pad
			else:
				tokens=tokens[:pad_len]
			# print (len(tokens),pad_len)
			assert len(tokens)==pad_len
		# print (tokens)
		# Convert token to vocabulary indices
		indexed_tokens = self.bt_tokenizer.convert_tokens_to_ids(tokens)
		return indexed_tokens,act_length
