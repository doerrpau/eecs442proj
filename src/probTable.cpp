// This impliments the EM Algorithm

int main()
{
	double smallChange = 10^(-4);
	double change = 10000;
	Mat probTable = ones(totNumWords, totNumBlobs,CV_32F);
	int iter,N,l,m = 0;
	double sumP,sumT=0;

	// Get words and blobs for each image
	N =  images.size()
	//========================= EM Algorithm ==================================//

	//============================ Init =======================================
	// Go over each image and fill in initial probabilities
	//FIX ME
	for(int n=0; n<N; n++)
	{
		for(int j=0;j<image[n].words.size();j++)
		{
			for(int i=0;i<images[n].blobs.size();i++)
			{
				images[n].pTable[j][i] = images[n].words.size()/images[n].blobs.size();
				images[n].tTablep[j][i] = 1/images[n].words.size();
			}
		}
	}
	//============================ End Init ====================================

	//================================ EM ======================================
	while(smallChange < change || iter <2)
	{
		//CONVERGENCE OR #ITER

		// E Step
		// Calculate p_tild(a_nj|w_nj,b_nj,old params) for each image over all
		// words and blobs

		for(n=0;n<N;n++)
		{
			sumP = 0;//for each image

			m = images[n].words.size();
			l = images[n].blobs.size();

			for(j=0;j<m;j++)
			{
				for(i=0;i<l;i++)
				{
					//for each word/blob
					pTemp[n][j][i] = images[n].pTable[j][i]*images[n].tTable[j][i];
					sumP+=pTemp[n][j][i];
				}
			}

			// Normalize p_tild(a_nj|w_nj,b_nj,old params)
			pTemp[n][j][i]/=sumP;
		}
		//============================ End E Step =================================//

		//============================== M Step ===================================//
		// | M.1 |
		// Get mixing probablilities by looking over each word for each image
		// of the same size

		//for each image set
		for(n=0;n<N;n++)
		{
			m = images[n].words.size();
			l = images[n].blobs.size();

			for(j=0;j<m;j++)
			{
				for(i=0;i<l;i++)
				{
					//Go over each image with same size
					countNlm = 0;
					sumP = 0;
					for(int nn =0;nn<N;nn++)
					{

						for(int jj=0;jj<images[nn].words.size();jj++)
						{
							if(images[nn].words.size()!=m)
							{
								break;
							}

							for(int ii=0;ii<images[nn].blobs.size();ii++)
							{
								//loop over conditions
								if(images[nn].blobs.size()!=l)
								{
									continue;
								}
								
								sumP+=pTemp[nn][jj][ii];	//if same number of words&blobs
								countNlm++;				//for each image in set
								
							}
						}
					}
					images[n].pTable[j][i] = sumP/countNlm;
				}
			}
			
		}

		// |M.2 & M.3|
		// Get t_tild(w_nj=w*|b_ni=b*) by looing for pairs (w*,b*) that appear
		// in same image...may just be unique assignment
		for(n=0;n<N;n++)
		{
			sumT=0;
			m = images[n].words.size();
			l = images[n].blobs.size();

			//for each word/blob pair
			for(j=0;j<m;j++)
			{
				for(i=0;i<l;i++)
				{

					sumP=0;
					//loop over all images to look for same blob pair
					for(int nn =0;nn<N;nn++)
					{
						for(int jj=0;jj<images[nn].words.size();jj++)
						{
							// See if word is in image
							if(images[nn].words[jj]!=images[n].words[j])
							{
								break;
							}

							for(int ii=0;ii<images[nn].blobs.size();ii++)
							{
								// See if blob is in image
								if(images[nn].blobs[jii]!=images[n].blobs[i])
								{
									continue;
								}

								sumP+=pTemp[nn][jj][ii];//if word&blob are in image
							}
						}
						images[n].tTable[j][i]=sumP;//at end of images loop
					}
					sumT+=sumP;
				}
			}

			//for each word/blob pair
			for(j=0;j<images[n].words.size();j++)
			{
				for(i=0;i<images[n].blobs.size();i++)
				{
					// Normalize t_tild(w_nj=w*|b_ni=b*) over each image
					images[n].tTable[j][i]/=sumT;
				}
			}
			
		}
		//================================ End M Step ===============================//
		iter++;
	}

	cout << "Change after iteration" <<change <<"\n";

	// Calculate final probability table
	double prod1,prod2 = 1;
	double sum1 = 0;

	for(int w=0;w<totNumWords;w++)
	{
		for(int b=0;b<totNumBlobs;b++)
		{
			prod1=1;
			for(int n=0;n<numImages;n++)
			{
				prod2=1;
				for(int j=0;j<images[n].words.size;j++)
				{
					sum1=0;
					for(int i=0;j<images[n].blobs.size;i++)
					{
						//If the blob b is in the image, add it to the sum
						// (it's zero otherwise)
						if(b = images[n].blobs[j])
						{
							sum1+=images[n].pTable[j][i]*images[n].tTable[j][i];
						}
					}

					prod2*=sum1;
				}
				prod1*=prod2;
			}
			probTable.at<double>[w][b]=prod1;	//Write to probTable matrix
		}
	}


	// Output probabilty table probTable to a file

	return 0;
}