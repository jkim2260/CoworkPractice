
# Plot(각 모델별로 그래프 그리기)

with open('Logistic Regression.pickle', 'wb') as f: #wb bite형식으로 저장
     pickle.dump(model, f)
     
#with open('multiple linear regression.pickle', 'wb') as f:
     #pickle.dump(model, f)

#with open('Ridge.pickle', 'wb') as f:
     #pickle.dump(model, f)

#with open('LASSO.pickle', 'wb') as f:
     #pickle.dump(model, f)

#with open('Elastic Net.pickle', 'wb') as f:
     #pickle.dump(model, f)
     
#with open('LARS.pickle', 'wb') as f:
     #pickle.dump(model, f)
     

with open("Logistic Regression.pickle", 'rb') as f:
    LR = pickle.load(f)

#with open("multiple linear regression.pickle", 'rb') as f:
    #MR = pickle.load(f)
    
#with open("Ridge.pickle", 'rb') as f:
    #RR = pickle.load(f)

#with open("LASSO.pickle", 'rb') as f:
    #Lasso = pickle.load(f)
    
#ith open("Elastic Net.pickle", 'rb') as f:
    #Elastic = pickle.load(f)
    
#with open("LARS.pickle", 'rb') as f:
    #LARS = pickle.load(f)
    
    
plt.figure(figsize=(15,10))

plt.plot(np.array(LR), 'g', label ='LR')
#plt.plot(np.array(MR),'k', label ='MR')
#plt.plot(np.array(RR),'y', label ='RR')
#plt.plot(np.array(Lasso),'r', label ='LASSO')
#plt.plot(np.array(Elastic),'r', label ='Elastic')
#plt.plot(np.array(LARS),'r', label ='LAR')


plt.title('Model loss')
plt.legend(loc='upper right')
plt.legend()
plt.grid('on')
plt.show()