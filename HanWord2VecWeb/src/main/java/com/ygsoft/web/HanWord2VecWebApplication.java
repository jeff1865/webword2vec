package com.ygsoft.web;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class HanWord2VecWebApplication {

	public static void main(String[] args) {
		System.out.println("Start Server .. ");
		SpringApplication.run(HanWord2VecWebApplication.class, args);
	}
}
