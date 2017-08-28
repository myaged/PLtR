package pltr;

/*
	PLtR-B
	Parallel pairwise learning to rank with block partioning
	See (Yagci et al., 2017)
	
	In this version:
	- Loss function and sampling are based on BPR-MF (Rendle et al., 2009)
	- i and j of (u,i,j) are sampled from the same chunk. j is sampled in worker threads.

	tested with Java 1.7
	@author: M. Yagci
 */
import java.util.*;

public class PLTRB implements PLTR {

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
	ArrayList<Tuple> data; // training data 
	Integer numProcs; // number of processors
	
	// Variables for partitioning step
	int[] usersShuffled; // array values start from 1
	int[] itemsShuffled; // array values start from 1
	List<List<Integer>> itemChunkMapping;
	List<List<List<Tuple>>> C;

	// -------------------------------------
	// Constructor
	// -------------------------------------
	public PLTRB(	int numUsers, 
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
	// generate permutations (using Fisher-Yates shuffle)
	// -------------------------------------	
	private void generatePerms(){
		int index, temp;
		Random random = new Random();
		
		for (int i = this.numUsers - 1; i > 0; i--){
		    index = random.nextInt(i + 1);
		    temp = this.usersShuffled[index];
		    this.usersShuffled[index] = this.usersShuffled[i];
		    this.usersShuffled[i] = temp;
		}
		
		for (int i = this.numItems - 1; i > 0; i--){
		    index = random.nextInt(i + 1);
		    temp = this.itemsShuffled[index];
		    this.itemsShuffled[index] = this.itemsShuffled[i];
		    this.itemsShuffled[i] = temp;
		}
		
		// extra function to decide mappings of item permutations to chunks
		itemChunkMapping = new ArrayList<List<Integer>>();
		for (int i = 0; i < this.numProcs; i++){
			itemChunkMapping.add(new ArrayList<Integer>());
		}
		for (int i = 0; i < this.numItems; i++){
			Integer b = (int)Math.floor( 1.0 * this.numProcs/this.numItems * (itemsShuffled[i]-1) ) + 1;
			itemChunkMapping.get(b-1).add(i);
		}		
	}

	// -------------------------------------
	// permute data
	// -------------------------------------	
	private void permuteData(){
	
		C = new ArrayList<List<List<Tuple>>>(); // initialize C anew
		for(int i=0; i<numProcs; i++) {
			C.add(new ArrayList<List<Tuple>>());
			for(int j=0; j<numProcs; j++) {
				C.get(i).add(new ArrayList<Tuple>());
			}
		}		

		// put data into partitions
		for (Tuple t : this.data){
		
			Integer userId = t.getUserId();
			Integer itemId = t.getItemId();
			
			Integer a = (int)Math.floor( 1.0 * this.numProcs/this.numUsers * (usersShuffled[userId]-1) ) + 1;
			Integer b = (int)Math.floor( 1.0 * this.numProcs/this.numItems * (itemsShuffled[itemId]-1) ) + 1;
			
			C.get(a-1).get(b-1).add(new Tuple(userId,itemId));			
			
		}		
	}

	// -------------------------------------
	// deciding set of blocks
	// -------------------------------------	
	private Integer blockDecider(Integer u, Integer a){
		Integer b = (u+a) % this.numProcs;
		if ( b==0 ){
			return this.numProcs;
		}
		return b;
	}
	
	// -------------------------------------
	// learn model
	// -------------------------------------
	public void learn(ArrayList<Tuple> data, Integer numProcs) throws InterruptedException{
	
		this.data = data;
		this.numProcs = numProcs;
	
		// build user histories in the first pass
		for( Tuple user_item : data){
			Integer user = user_item.getUserId();
			Integer item = user_item.getItemId();
			
			if( ! BPlus.containsKey(user) ){
				BPlus.put(user, new HashSet<Integer>());
			}
			BPlus.get(user).add(item);
		}
		
		// initial filling for in-place shuffle
		usersShuffled = new int[this.numUsers];
		for( int i=1; i < this.numUsers+1; i++ ){
			usersShuffled[i-1] = i; // fill
		}
		itemsShuffled = new int[this.numItems];	
		for( int i=1; i < this.numItems+1; i++ ){
			itemsShuffled[i-1] = i; // fill
		}
		
		// run epochs
		for (int epoch=0; epoch<this.numEpochs; epoch++){	
			System.out.println("epoch: " + epoch);
			
			this.generatePerms();
			this.permuteData();			
		
			// parallel processing coordination by deciding set of blocks
			ArrayList<Thread> threadList;
			final Integer numProcsFinal = numProcs;
			for (int u=0; u<this.numProcs; u++){
				threadList = new ArrayList<Thread>();
				for (int a=0; a<this.numProcs; a++){
					final int aa = a;
					final int b = blockDecider(u,aa+1);
				
					threadList.add ( new Thread(new Runnable() {
						public void run(){
							long threadId = Thread.currentThread().getId() % numProcsFinal;
							updateParallel(aa, b-1, threadId);
						}
					}) );
					threadList.get(a).start(); 		
				}
				for(int i=0; i<this.numProcs; i++){
					threadList.get(i).join();		
				}
			}
		}		
	}
	
	// -------------------------------------
	// -------------------------------------
	private void updateParallel(int a, int b, long id){
	
		// System.out.println("Running thread " + id + " ...");
		
		// update model
		for (Tuple t : C.get(a).get(b)){
		
			Integer user = t.getUserId();
			Integer posItem = t.getItemId();
			
			// Integer negItem = t.getItemId2();
			// sample negative item
			Integer negItem = -1;
			Integer numTrials = 0;
			Integer itemChunkMappingSize = itemChunkMapping.get(b).size();			
			while( numTrials < 10 ){
				Integer rnd = java.util.concurrent.ThreadLocalRandom.current().nextInt(0, itemChunkMappingSize);
				Integer rndItem = itemChunkMapping.get(b).get(rnd);
				if( ! BPlus.get(user).contains(rndItem) ){
					negItem = rndItem;
					break;
				}
				numTrials += 1;
			}
			if( negItem == -1 ){
				continue;
			}
		
			Double delta = 1.0 - sigmoid( MatrixOps.diffDot(P[user], Q[posItem], Q[negItem]) );

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

	// -------------------------------------
	// -------------------------------------
	private Double sigmoid(Double x){
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

