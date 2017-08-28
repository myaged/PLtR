package pltr;

/*
	Example tester file for PLtR
	
	*** This software is provided "as is",
	and for research purposes. ***

	tested with Java 1.7
	@author: M. Yagci
 */

import java.util.*;
import java.io.*;

public class Tester {

	public static void main(String[] args) {
        
        // ------------------------------------
		// User input parameters
		// ------------------------------------

		// Training file 
		// (Tab-delimited. Columns: <user,item,...>. Ids: [0,max. id of users/items])
		String trainingFile = "../data/train_data_format_example.csv";

		// BPR parameters
		Integer numLatentFactors = 20;
		Double mu = 0.0;
		Double sigma = 0.01;
		Double lambP = 0.0025;
		Double lambQPlus = 0.0025;
		Double lambQMinus = 0.00025;
		Double eta = 0.01;
		
		String algorithm = "PLTRN"; // Choose: "SEQ", "PLTRN", or "PLTRB"
		
		Integer numEpochs = 4; // PltR is expected to run in numEpochs/numProcs units of time	
		Integer numProcs = 4; // >1: some parallelism, 1: no parallelism (neglected in SEQ)
		
		// ------------------------------------
		// Reading training data to MM
		// ------------------------------------
		System.out.println("Reading training data ...");

		Integer maxUserId = 0;
		Integer maxItemId = 0;
		ArrayList<Tuple> trainData = new ArrayList<Tuple>();

		try {
			BufferedReader reader = new BufferedReader(new FileReader(trainingFile));
			String row;
			row = reader.readLine();			
			while ((row = reader.readLine()) != null) {
				String[] fields = row.split("\t");
				Integer userId = Integer.parseInt(fields[0]);
				Integer itemId = Integer.parseInt(fields[1]);	
				
				if (userId > maxUserId)
					maxUserId = userId;
				if (itemId > maxItemId)
					maxItemId = itemId;
				
				Tuple newTuple = new Tuple(userId,itemId);
				trainData.add(newTuple);			
			}
			reader.close();
		} catch (IOException e) {
			System.out.println("File Read Error");
		}

		// ------------------------------------
		// Training
		// ------------------------------------
		System.out.println("initializing and learning model ...");			
		
		PLTR model;		
		switch (algorithm) {
			case "SEQ":
				model = new BPR(maxUserId+1, maxItemId+1, numLatentFactors, mu, sigma, lambP, lambQPlus, lambQMinus, eta, numEpochs);
				break;
			case "PLTRN":
				model = new PLTRN(maxUserId+1, maxItemId+1, numLatentFactors, mu, sigma, lambP, lambQPlus, lambQMinus, eta, numEpochs);
				break;
			case "PLTRB":
				model = new PLTRB(maxUserId+1, maxItemId+1, numLatentFactors, mu, sigma, lambP, lambQPlus, lambQMinus, eta, numEpochs);
				break;
			default:
				throw new IllegalArgumentException("Invalid algorithm !!!");
		}
	
		long startTime = System.nanoTime();		
			try {
				model.learn(trainData, numProcs);
			}
			catch(InterruptedException e) {
				System.out.println("Threading error ...");
			}		
		System.out.println("it took (secs): "+ (System.nanoTime() - startTime) / 1e9);
	}
	
}

