package pltr;

/*
	BPR-MF algorithm 
	see (Rendle et al., 2009)

	tested with Java 1.7
	@author: M. Yagci
 */
import java.util.*;

public class BPR implements PLTR {

	int numUsers;
	int numItems;
	int numLatentFactors;
	double[][] P; // user component matrix
	double[][] Q; // item component matrix
	double lambP; // regularization parameter
	double lambQPlus; // regularization parameter
	double lambQMinus; // regularization parameter
	double eta; // learning rate
	int numEpochs; // number of training epochs
	Map<Integer,Set<Integer>> BPlus; // user histories

	// -------------------------------------
	// Constructor
	// -------------------------------------
	public BPR(	int numUsers, 
				int numItems,
				int numLatentFactors, 
				double mu,
				double sigma,
				double lambP,
				double lambQPlus,
				double lambQMinus,
				double eta,
				int numEpochs) {
		
		this.numUsers = numUsers;
		this.numItems = numItems;
		this.numLatentFactors = numLatentFactors;
		this.P = MatrixOps.gaussianMatrixBuilder(mu, sigma, numUsers, numLatentFactors);
		this.Q = MatrixOps.gaussianMatrixBuilder(mu, sigma, numItems, numLatentFactors);
		this.lambP = lambP;
		this.lambQPlus = lambQPlus;
		this.lambQMinus = lambQMinus;
		this.eta = eta;
		this.numEpochs = numEpochs;
		this.BPlus = new HashMap<Integer,Set<Integer>>(); 
	}

	// -------------------------------------
	// learn model (with uniform sampling)
	// -------------------------------------
	public void learn(ArrayList<Tuple> data, Integer numProcs) throws InterruptedException{
		
		// build user histories in the first pass
		for( Tuple user_item : data){
			Integer user = user_item.getUserId();
			Integer item = user_item.getItemId();
			
			if( ! BPlus.containsKey(user) ){
				BPlus.put(user, new HashSet<Integer>());
			}
			BPlus.get(user).add(item);
		}
		
		// update model
		int epoch = 0;
		int lenData = data.size();
		for( int i=0; i<this.numEpochs; i++ ){
			System.out.println("epoch: " + epoch);
			for( int j=0; j<lenData; j++ ){
				// sample with repetition
				Integer rnd = java.util.concurrent.ThreadLocalRandom.current().nextInt(0, lenData);
				Integer user = data.get(rnd).getUserId();
				Integer posItem = data.get(rnd).getItemId();
				Integer negItem = -1;
				int numTrials = 0;
				while( numTrials < 10 ){
					Integer rnd2 = java.util.concurrent.ThreadLocalRandom.current().nextInt(0, this.numItems);
					if( ! BPlus.get(user).contains(rnd2)){
						negItem = rnd2;
						break;
					}
					numTrials += 1;
				}
				if( negItem != -1 ){
					double delta = 1.0 - sigmoid( MatrixOps.diffDot(P[user], Q[posItem], Q[negItem]) );
		
					for(int f=0; f<this.numLatentFactors; f++){
						P[user][f] += this.eta * 
						(delta * (Q[posItem][f] - Q[negItem][f]) - this.lambP * P[user][f]);
					}
		
					for(int f=0; f<this.numLatentFactors; f++){
						Q[posItem][f] += this.eta * 
						(delta * P[user][f] - this.lambQPlus * Q[posItem][f]);
					}
		
					for(int f=0; f<this.numLatentFactors; f++){
						Q[negItem][f] += this.eta * 
						(delta * -1.0*P[user][f] - this.lambQMinus * Q[negItem][f]);
					}
				}
			}
			epoch += 1;
		}		
	}

	// -------------------------------------
	// -------------------------------------
	private double sigmoid(double x){
		if( x > 0 ){
			return 1.0 / (1.0 + Math.exp(-x));
		} else if (x <= 0) {
			return Math.exp(x) / (1.0 + Math.exp(x));
		} else {
			System.out.println("Sigmoid value error ...");
			return 0.0;
		}
	}
	
	// -----------------------------------------------
	// evaluate a single (user,item) test pair for AUC
	// -----------------------------------------------
	public double getAUCUserItem(Integer user, Integer item){
	
		Double estimatedRankingScoreItem = MatrixOps.dot(P[user],Q[item]);
		Integer numNoInversions = 0;
		Integer numEligibleItems = 0;
		
		for(int i=0; i<this.numItems; i++){
			if( ( ! BPlus.get(user).contains(i) ) && item != i ) {
				numEligibleItems++;
				if ( estimatedRankingScoreItem > MatrixOps.dot(P[user],Q[i]) ) {
					numNoInversions++;
				}
			}		
		}
		
		return 1.0 * numNoInversions / numEligibleItems;
	}
	
}

