import torch
from torch.autograd import Variable

def adversarial_dropout(cur_mask, Jacobian, dim, change_limit, name="ad"):
    
    cur_mask = torch.gt( cur_mask, torch.zeros(cur_mask.size()).cuda() ).float()
    changed_mask = cur_mask.view(-1, dim)
    
    if change_limit > 1 :
        dir = Jacobian.view(-1, dim)
        temp_ones = torch.ones(changed_mask.size()).cuda()
        m = (2.0*changed_mask - temp_ones)
        
        # remain (J>0, m=-1) and (J<0, m=1), which are candidates to be changed
        change_candidate = (temp_ones - torch.sign(dir*m).float() )/2                
        
        ads_dL_dx = torch.abs(dir)
        
        # ordering abs_Jacobian for candidates
        # the maximum number of the changes is "change_limit"
        # draw top_k elements ( if the top k element is 0, the number of the changes is less than "change_limit" ) 
        left_values = - change_candidate*ads_dL_dx
        
        kth_largest_values, kth_largest_idxes = torch.kthvalue(left_values.cpu(), change_limit, dim=1) 
        kth_largest_values = kth_largest_values.view(-1, 1).cuda()
        change_target = torch.lt(left_values, kth_largest_values.expand_as(left_values) ).float()
        
        # changed mask with change_target
        
        changed_mask =  (m - 2.0*m*change_target + temp_ones)*0.5
        num_act = torch.sum(changed_mask, dim=1, keepdim=True)
        changed_mask = changed_mask*float(dim)/num_act
    
    changed_mask = changed_mask.view( cur_mask.size()).clone()
    
    return changed_mask