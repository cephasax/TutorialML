package br.ufrn.imd.tutorial;

import com.fasterxml.jackson.annotation.JsonProperty;

public class Iris {

	/**
	 * @ATTRIBUTE sepallength REAL
	 * @ATTRIBUTE sepalwidth REAL
	 * @ATTRIBUTE petallength REAL
	 * @ATTRIBUTE petalwidth REAL
	 * @ATTRIBUTE class {Iris-setosa,Iris-versicolor,Iris-virginica}
	 */

	private double sepallength;
	private double sepalwidth;
	private double petallength;
	private double petalwidth;
	
	@JsonProperty("class")
	private String irisClass;

	public Iris() {
		this.irisClass = new String("Iris-setosa");
	}

	public double getSepallength() {
		return sepallength;
	}

	public void setSepallength(double sepallength) {
		this.sepallength = sepallength;
	}

	public double getSepalwidth() {
		return sepalwidth;
	}

	public void setSepalwidth(double sepalwidth) {
		this.sepalwidth = sepalwidth;
	}

	public double getPetallength() {
		return petallength;
	}

	public void setPetallength(double petallength) {
		this.petallength = petallength;
	}

	public double getPetalwidth() {
		return petalwidth;
	}

	public void setPetalwidth(double petalwidth) {
		this.petalwidth = petalwidth;
	}

	public String getIrisClass() {
		return irisClass;
	}

	public void setIrisClass(String irisClass) {
		this.irisClass = irisClass;
	};

}
