package pltr;

/*
	Tuple class

	tested with Java 1.7
	@author: M. Yagci
 */
public class Tuple {

	Integer userId;
	Integer itemId;

	public Tuple(Integer userId, Integer itemId) {
		this.userId = userId;
		this.itemId = itemId;
	}

	public Integer getUserId(){
		return userId;
	}

	public Integer getItemId(){
		return itemId;
	}
}
