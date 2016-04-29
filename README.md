#The Information Sieve

Recent work establishes results that allow the efficient construction of representations that are *maximally informative* 
about the data. A recent extension allows us to build maximally
informative representations in an incremental way. The other advantage of this approach is that it keeps track of
 *remainder information*, so that we can enable lossy and lossless compression. 
  
This is described here: 
*The Information Sieve*
Greg Ver Steeg and Aram Galstyan, ICML 2016, http://arxiv.org/abs/1507.02284

Papers describing previous work:     
*Discovering Structure in High-Dimensional Data Through Correlation Explanation*    
Greg Ver Steeg and Aram Galstyan, NIPS 2014, http://arxiv.org/abs/1406.1222        
      
*Maximally Informative Hierarchical Representions of High-Dimensional Data*    
Greg Ver Steeg and Aram Galstyan, AISTATS 2015, http://arxiv.org/abs/1410.7404    

###Dependencies

Sieve requires numpy and scipy. If you use OS X, I recommend installing the Scipy Superpack:             
http://fonnesbeck.github.io/ScipySuperpack/

###Install

To install, download using the link on the right or clone the project by executing this command in your target directory:
```
git clone https://github.com/gregversteeg/sieve.git
```
Use *git pull* to get updates. The code is under heavy development. 
Please feel free to raise issues or request features using the github interface. 

## Basic Usage

### Example

```python
import corex as ce

X = np.array([[0,0,0,0,0], # A matrix with rows as samples and columns as variables.
              [0,0,0,1,1],
              [1,1,1,0,0],
              [1,1,1,1,1]], dtype=int)

s = sieve.Sieve(max_layers=2)  # Initialize. Can limit number of layers (default is automatic) and pass kwargs to CorEx
s.fit(X)

s.labels  # Each variable/column is associated with one Y_j
s.tcs  # TC(X; Y) for each latent factor
s.layers[0].corex  # Access the CorEx object 

```

More examples are in the paper and in the tests and paper_figures folder. 

### CorEx outputs

See https://github.com/gregversteeg/CorEx for more info about the CorEx module (the one 
included here is a simplified version.)