import torch as t
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import LoadDataset
from GCNv import GCN
import cfg0
from metrics import averageMeter, runningScore
import time
import os.path as osp
device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
num_class = cfg0.DATASET[1]


pth = "Set the path to your saved model file"
Load_test = LoadDataset([cfg0.TEST_ROOT, cfg0.TEST_LABEL], cfg0.crop_size)
test_data = DataLoader(Load_test, batch_size=cfg0.BATCH_SIZE, shuffle=False, num_workers=4)

net = GCN(num_class)
net.load_state_dict(t.load(osp.join(cfg0.Weight_Path,'{}'.format(pth))) )
net.eval()
net.to(device)
running_metrics_val = runningScore(n_classes=num_class)
time_meter = averageMeter()

for i, sample in enumerate(test_data):
	time_start = time.time()
	data = Variable(sample['img']).to(device)
	label = Variable(sample['label']).to(device)
	out = net(data)
	out = F.log_softmax(out, dim=1)

	pre_label = out.max(dim=1)[1].data.cpu().numpy()
	true_label = label.data.cpu().numpy()
	running_metrics_val.update(true_label, pre_label)
	time_meter.update(time.time() - time_start, n=data.size(0))

metrics = running_metrics_val.get_scores()

for k, v in metrics[0].items():
	print(k, v)
print(metrics[1])
print('inference time per image: ', time_meter.avg)
print('inference fps: ', 1 / time_meter.avg)