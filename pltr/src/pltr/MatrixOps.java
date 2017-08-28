package pltr;

/*
	Matrix operations utility

	tested with Java 1.7
	@author: M. Yagci
 */
import java.util.*;

public class MatrixOps {

	// -------------------------------------
	// gaussian random matrix builder
	// -------------------------------------
	public static double[][] gaussianMatrixBuilder(	double mu,
													double sigma,
													int numRows,
													int numCols ){
	
		double[][] matrix = new double[numRows][numCols];
		for(int i=0; i<numRows; i++){
			for(int j=0; j<numCols; j++){
				matrix[i][j] = 
				java.util.concurrent.ThreadLocalRandom.current().nextGaussian() 
				* sigma + mu;			
			}
		}
		return matrix;		
	}
	
	// -------------------------------------
	// dot product of two vectors
	// -------------------------------------
	public static double dot(double[] x, double[] y){		
		double dotProduct = 0.0;
		for( int i=0; i<x.length; i++){
			dotProduct += x[i]*y[i];
		}		
		return dotProduct;
	}
	
	// -------------------------------------
	// difference of dot products
	// -------------------------------------
	public static double diffDot(double[] x, double[] y, double[] z){		
		double diffDotProduct = 0.0;
		for( int i=0; i<x.length; i++){
			diffDotProduct += x[i]*(y[i]-z[i]);
		}		
		return diffDotProduct;
	}
	
}

