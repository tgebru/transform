'''Automate experiments to train networks for inversions for 
   summer-->summer
   summer-->winter
   winter-->summer 
'''
import os
import subprocess as subp
import glob

#Incase I want to experiment with changing these values
lr='0.00000000001'
gammas=['0.1','0.3','0.5','1']
stepsizes= ['1000', '3000', '5000','10000']
weight_decays= ['0','0.0005', '0.005', '0.05']
momentum= '0.9'
max_iter= 10000

#Best parameters based on before
stepsize=stepsizes[-1]
weight_decay=weight_decays[2]
gamma=gammas[2]

exps=['summer_winter','winter_summer','same']
size=['tiny','small','big']

#no_of_exps=len(gammas)*len(stepsizes)*len(weight_decays)
no_of_exps=len(exps)*len(size)

exp_no=0
#Run experiments one by one
#for gamma in gammas:
#  for weight_decay in weight_decays:
#    for stepsize in stepsizes:
for s in size:
  if s=='big':
    max_iter=50000
  else:
    max_iter=10000
  for exp in exps:
    exp_no += 1
    print 'running exp %d out of %d'%(exp_no,no_of_exps)
    base_dir='/scr/r6/tgebru/inverting_conv/caffe_invert_alexnet/timnit_models/att_conv1'
    solver_file=os.path.join(base_dir,'solver.prototxt')
    f=open(solver_file,'rb')
    line=f.readlines()
    f.close()
    f=open(solver_file,'w')
    ltr='train_net: "%s/%s/train_%s.prototxt"\n'%(base_dir,exp,exp)
    lt='test_net: "%s/%s/val_%s.prototxt"\n'%(base_dir,exp,exp)
    if not os.path.exists("%s/%s/%s/snapshot"%(base_dir,exp,s)):
      os.makedirs("%s/%s/%s/snapshot"%(base_dir,exp,s))
    ls='snapshot_prefix: "%s/%s/%s/snapshot/%s.%s.%s.%s.%s.%s.%s"\n'%(base_dir,exp,s,lr,gamma,stepsize,momentum,weight_decay,max_iter,exp)
    i=0
    #Change solver file
    for l in line:
      if l.startswith('train_net'):
        l=ltr
      if l.startswith('test_net'):
        l=lt
      elif l.startswith('snapshot_prefix'):
        l=ls
      elif l.startswith('max_iter'):
        l='max_iter: %d\n'%max_iter
      f.write(l)
    f.close()

    #Change source and target images on prototxt file and batchsize
    ltr=ltr.split(':')[1].strip().replace('"','')
    lt=lt.split(':')[1].strip().replace('"','')
    print ltr
    print lt
    ftr=open(ltr,'rb')
    ft=open(lt,'rb')
    trainlines=ftr.readlines()
    testlines=ft.readlines()
    ftr.close()
    ft.close()

    ftr=open(ltr,'w')
    ft=open(lt,'w')
    for tr,ts in zip(trainlines,testlines):
      if tr.strip().startswith('source:'): 
        #Change train data file to append 'tiny,small, etc...'
        partstr=tr.split('/')
        fname=partstr[-1]
        parts_fname=fname.split('_')
        if len(parts_fname)==4:
          parts_fname[0]=s
        else:
          parts_fname.insert(0,s)
        new_fname='_'.join(parts_fname)
        partstr[-1]=new_fname
        tr='/'.join(partstr)
        
        #Do the same thing for the test data file
        partstr=ts.split('/')
        fname=partstr[-1]
        parts_fname=fname.split('_')
        if len(parts_fname)==4:
          parts_fname[0]=s
        else:
          parts_fname.insert(0,s)
        new_fname='_'.join(parts_fname)
        partstr[-1]=new_fname
        ts='/'.join(partstr)

      #Change the batchsize for small/tiny
      if s != 'big':
         train_batchsize=12
         test_batchsize=2
      else:
         train_batchsize=64
         test_batchsize=64
        
      if tr.strip().startswith('batch_size'):
        tr='batch_size: %d'%train_batchsize
        ts='batch_size: %d'%test_batchsize
    
      ftr.write(tr)
      ft.write(ts)
    ftr.close()
    ft.close()
    print 'Fintunning...'
    subp.call(["bash", "%s"%os.path.join(base_dir,'finetune.sh')])

    #Copy the final log file to the scripts section and run next experiment
    print 'Copying log file..'
    log=max(glob.iglob('/tmp/finetune_net.bin.tibet2.tgebru.log.*'),key=os.path.getctime)
    subp.call(["cp", "%s"%log, "."])
    subp.call(["mv", "%s"%log,"%s.%s.%s.%s.%s.%s.%s.%s.log"%(lr,gamma,stepsize,momentum,weight_decay,max_iter,exp,s)])

