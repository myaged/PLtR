package pltr;

/*
	Simple interface for algorithms

	tested with Java 1.7
	@author: M. Yagci
 */
import java.util.*;

public interface PLTR {

	public void learn(ArrayList<Tuple> data, Integer numProcs)  throws InterruptedException;
	public double getAUCUserItem(Integer user, Integer item);
	
}
