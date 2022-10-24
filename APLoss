class Finegrainedloss4(nn.Module):

    def __init__(self, in_features, out_features, s=64, m1=0.35, m2=2,keep_num=5):
        super(Finegrainedloss4, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.keep_num = keep_num
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.w = nn.Parameter(torch.ones(1))

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        cmp = torch.zeros(cosine.shape[0], device='cuda:0')
        for i in range(cosine.shape[0]):
            cmp[i] = abs(cosine[i][0]-cosine[i][1])
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda:0')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output2 = torch.zeros(cosine.size(), device='cuda:0')
        for i in range(cosine.shape[0]):
            phi = cosine[i] - (1/ (1 + math.exp(1.6 * cmp[i])))
            output2[i] = (one_hot[i] * phi) + ((1.0 - one_hot[i]) * cosine[i])
        output2 *= self.s
        return output2
