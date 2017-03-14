/*****************************************************************************************/
/**********																		**********/
/**********								Eureka.c								**********/
/**********					Otimização de Mecanismos Flexíveis					**********/
/**********																		**********/
/**********						Por Vitor Bellissimo Falleiros					**********/
/**********				para obtenção do título de Engenheiro Mecânico			**********/
/**********						pela Escola Politécnica da USP					**********/
/**********									2003								**********/
/**********																		**********/
/*****************************************************************************************/



#include <stdio.h>
#include <math.h>
#include <string.h>
#include <malloc.h>
#include <stdlib.h>

#define entrada1	"carga1.cdb"
#define entrada2	"carga2.cdb"	// Arquivos de entrada do modelo e dos carregamentos

#define MATLAB		"iter.m"
#define ANSYS		"result.inp"	// Arquivos de saida com evolução do processo e resultado
#define A0			"r0.inp"		// Arquivos de saida com evolução do processo e resultado
#define A1			"r1.inp"		// Arquivos de saida com evolução do processo e resultado
#define A2			"r2.inp"		// Arquivos de saida com evolução do processo e resultado
#define A3		    "r3.inp"		// Arquivos de saida com evolução do processo e resultado
#define A4			"r4.inp"		// Arquivos de saida com evolução do processo e resultado
#define A5			"r5.inp"		// Arquivos de saida com evolução do processo e resultado
#define A6			"r6.inp"		// Arquivos de saida com evolução do processo e resultado
#define A7			"r7.inp"		// Arquivos de saida com evolução do processo e resultado
#define A8			"r8.inp"		// Arquivos de saida com evolução do processo e resultado
#define A9			"r9.inp"		// Arquivos de saida com evolução do processo e resultado

#define NMAX            2.0e7		// Tamanho dos vetores de alocação esparsa p/ matrix de rigidez
#define TAM				200			// Tamanho máximo do string a ser lido dos arquivos a cada vez
#define CONT			3.0e-2		// tolerância de troca da penalização (critério da continualidade)
#define ITERTOL			1.0e-2		// tolerância no resultado da otimização
#define ITMAXLBCG		200			// Limite de iterações da função linbcg
#define MAXITER			201			// limite de iterações geral
#define TOL				5.0e-3		// tolerância na linbcg
#define EPS				1.0e-6		// Número infinitesimal
#define PEN1			1			//
#define PEN2			3			// penalização para baixas densidades
#define w1				0.1			// para construção da função objetivo
#define w2				0.1			// para construção da função objetivo
#define w3				0.4			// para construção da função objetivo
#define w4				0.4			// para construção da função objetivo
#define NR_END          1			// Parâmetro para funções de alocação
#define FREE_ARG        char*		// Parâmetro para funções de alocação
#define RMAX			1.5e-2		// raio de abrangência do filtro
#define minf			0.95		// variação dos limites móveis
#define msup			1.05
#define mllower			0.04		// limites pros limites móveis
#define mlupper			0.15


/***************************************************************************************/
/**************** Header das funções usadas ao longo do programa ***********************/
/***************************************************************************************/

/* Funções do Numerical Recipes "nrutil" */
// Dá mensagem de erro e encerra programa:
void nrerror(char error_text[]);
// Funções de alocação de memória:
float *vector(long nl, long nh);
int *ivector(long nl, long nh);
unsigned char *cvector(long nl, long nh);
unsigned long *lvector(long nl, long nh);
double *dvector(long nl, long nh);
float **matrix(long nrl, long nrh, long ncl, long nch);
double **dmatrix(long nrl, long nrh, long ncl, long nch);
int **imatrix(long nrl, long nrh, long ncl, long nch);
// coloca uma matriz finita dentro de outra, talvez infinita:
float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch,long newrl, long newcl);
// Converte matriz normal em ponteiro (indexada?):
float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch);
// Aloca memória p/ matriz 3d:
float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh);
// Liberam memória das variáveis:
void free_vector(float *v, long nl, long nh);
void free_ivector(int *v, long nl, long nh);
void free_cvector(unsigned char *v, long nl, long nh);
void free_lvector(unsigned long *v, long nl, long nh);
void free_dvector(double *v, long nl, long nh);
void free_matrix(float **m, long nrl, long nrh, long ncl, long nch);
void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch);
void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch);
void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch);
void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch);
void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,	long ndl, long ndh);


/* Funções para trabalho com matriz esparsa e resolução de sistema montado com matriz esparsa */

// soma valor na matriz esparsa representada por sa e ija
void insere(int i, int j, double v);
// Prepara vetores ija e sa para montagem da matriz global
void Prepara(long ksize, int **vizinhos);
// Resolve A*x=b pelo método dos gradientes biconjugados lineares (?):
void linbcg(unsigned long n, double b[], double x[], double tol, int itmax, int *iter, double *err);
// Multiplica A por vetor x:
void dsprsax(double sa[], unsigned long ija[], double x[], double b[], unsigned long n);
// Multiplica A transposta por vetor x:
void dsprstx(double sa[], unsigned long ija[], double x[], double b[], unsigned long n);
// Resolve Ã*x=b, onde Ã é uma matriz pré-condicionada para facilitar a conta:
void asolve(unsigned long n, double b[], double x[], int itrnsp);
// Módulo de sx ou da maior componente dele:
double snrm(unsigned long n, double sx[]);


/* Funções de apoio ao programa de MEF */
// Posiciona o ponteiro de arquivo para letura dos dados desejados:
void posiciona(char **temp, FILE **arq1, char *desejado, int comprimento);
 // monta matriz de rigidez global nos vetores externos sa e ija
void MontaGlobal(long ksize,int M,int N,int *LM,int **vizinhos,double *x,double p,float ***elementos);
// Calcula distancia ente centros de 2 elementos
float distancia(float x1, float y1, float z1, float x2, float y2, float z2);
// Calcula gradiente:
float gradH8(float *xe,float *ye,float *ze,float r,float s,float t,float *N,float J,float *dNdx,float *dNdy,float *dNdz);
void inversa(float **A, float **B);  // inverte matriz
float determinante(float **B);  // calcula determinante de matriz 3x3
void multiplica(float **A, float **B, float **B2, float **ske, float J);  // J*[(B2xA)xB]
void multiplica2(float **A, float **B, float **B2, float **ske, float J);  // 4*J*[(B2xA)xB]
// Calcula e monta matriz de rigidez local
void SKE(int el, int *conect, float *coord,double E,double v, float **ske, int NE);
void transposta(float **B, float **B2);  // Transpõe matriz 6x4
void MM(float **A, float **B, int iA, int jA, int iB, int jB, float **C);  // Multiplica 2 matrizes normais

/* Funções para otimização */
// Função que atualiza as variáveis pelo critério da optimalidade e retorna o volume otimizado
double OC(double *x,double *xnew,double *grad,double *xlow,double *xup,float v,int n);
// Função que calcula os limites móveis
void limites(int M, double *sign1, double *sign2, double *sign3, double *ml,	double *x, double *xupper, double *xlower);
// função que aplica o filtro
void filtro(int M, float *cx, float *cy, float *cz, double *bu, double *bl, double radius);


/* Variaves globais */
unsigned long *ija;
double *sa;


/* Inicio do Main */
void main()
{

	/* declaraçao de variaveis */

	// Contadores e flags
	int opcao, op2, iter, it;
	int i, j, k;			// Contadores
	unsigned long uflag;

	double penal;

	// Leitura de dados
	char *temp;  // Armazena a string lida no momento
	FILE *arq1;  // Arquivo do AnSys com o modelo do domínio
	// Número de nós e elementos
	int NE, M;  // Número de nós e de elementos
	char c1[14];  // auxiliar para leitura de arquivo
	// coordenadas dos nós:
	int AuxCoord1, AuxCoord2, AuxCoord3;   // auxiliares para leitura das coordenadas
	float CoordT1, CoordT2, CoordT3;     // Coordenadas (temporários)
	float *coord;  // Todas as coordenadas X em sequência, depois todas as Y, depois todas as Z
	// Conectividade
	int b1, b2, b3, b4, b5, b6, b7, b8, b9;  // auxiliares de leitura de conectividade dos nós
	int ElemAtual, No1, No2, No3, No4, No5, No6, No7, No8;  // conectividade de cada elemento
	int *conect; // Matriz de conectividade, oito nós por elemento
    float *cx, *cy, *cz;  // Vetor com o centro geométrico de cada elemento, para usar no filtro (xadrez)
	// Material
	char c3[15], c4[5], c5, c6[5], c7[5]; // auxiliares p/ leitura da elasticidade
	char g1[15], g2[10], g3[5], g4[5];  // auxiliares p/ leitura do coef de poisson
	double E, V;  // características do material
	// restrições dos nós:
	unsigned long d2;  // nó, elemento ou dof analisado no momento;
	int *spc12, *spc3, *spc4;  // Matrizes c/ restrições de movimento (x1, y1, z1, x2, y2, z2, x3.....xNE, yNE, zNE)
	char d1[5], d3[5];  // auxiliares
	char e1, e2, e3;  // dofs restritos
	// esforços nos nós:
	float valor_atual;  // valor do esforço
	char direcao_atual; // direção do esforço
	char AuxF1[5], AuxF2[5], AuxF3[5], AuxF4, AuxF5, AuxF6;  // auxiliares para leitura das forças
	double *carga14, *carga23;  // matriz com o valor de cada esforço

	// Matriz reduzida
	int *ID12, *ID3, *ID4;  // Enumera os DOFs irrestritos na mesma sequencia lida em 'spc?'; DOFs restritos recebem zero
	unsigned long ksize12, ksize3, ksize4;  // Tamanho da matriz reduzida (sem os dofs restritos)
	// esforços da matriz reduzida
	double *f1, *f2, *f3, *f4;  // valores dos esforços não nulos
	// "Conectividade dos elementos"
	int *edof;  // auxiliar
	int *LM12, *LM3, *LM4;	// Conectividade dos elementos em relação aos DOFs livres

	// Otimização
	float x0;	// densidade inicial dos elementos
	double *xnew, *xold;	// Densidades: após iteração, antes da iteração
	float Vfrac;  // Limite de volume da estrutura final
	int **viz12, **viz3, **viz4;            // dofs que compartilham um mesmo elemento
	float ***mre;  // Todas as matrizes locais
	double *u1, *u2, *u3, *u4, uea[24], ueb[24];  // Deslocamentos globais e locais
    double *bf12, *ubif12, *bf3, *ubif3, *bf4, *ubif4; // Deslocamentos p/ linbcg
	double err;  // Recebe o erro máximo de linbcg
	double objetivold, *objetivo, *difer;  // Função objetivo na iter anterior e por iter, e variação por iter
	double *EnMut, *GradEM, *GradObjetivo; // Energia mútua, gradiente e gradiente da objetivo
	double *FM11, *Grad11, *FM22, *Grad22, *FM33, *Grad33, *FM44, *Grad44;	// Flexibilidade média e gradientes
	double tempmat[24];							// Auxiliar p/ calcular gradientes
	double *volume;                 // volume total da estrutura

	// limites móveis e filtro
	double *xupper, *xlower;
	double *sinal1, *sinal2, *sinal3, *ml;
	// Saída de dados
	FILE *arquivo;  // Arquivo de MatLab que gera gráficos sobre o processo iterativo
	FILE *arq3;     // Arquivo de AnSys que gera o modelo da estrutura final

/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////INICIO DA LEITURA DE DADOS///////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////
/*PROCEDIMENTO:	1.MODELAGEM NO ANSYS DAS 2 SITUAÇÕES DE CARREGAMENTO (esforços de entrada
								 e saída do mecanismo)
		2.GERAR ARQUIVOS CDWRITE NO ANSYS                               */
/////////////////////////////////////////////////////////////////////////////////////

// Leitura de dados que valem pra todos os casos (malha e material):

/* Abertura do arquivo .cdb */

	arq1=fopen(entrada1,"r");

	temp = (char *)malloc(TAM*sizeof(char ));

//////////////////////////////////////////////////////////////////////////////////////

/* VALOR DE "NE" (NUMERO DE NOS) */

	posiciona(&temp, &arq1, "NUMOFF,NODE,", 12);

	sscanf( temp, "%s %d", c1, &NE ); /*le a string do temp e atribui a N como inteiro*/

        printf("O valor de NE (numero de nos) eh: %d\n", NE);

////////////////////////////////////////////////////////////////////////////////////

/* VALOR DE "M" (NUMERO DE ELEMENTOS) */

	fgets(temp, TAM, arq1);
	sscanf( temp, "%s %d", c1, &M ); /*le a string do temp e atribui a M como inteiro*/

	printf("O valor de M (numero de elementos) eh: %d\n", M);

/////////////////////////////////////////////////////////////////////////////////////

/* VETOR "COORD" (COORDENADAS DOS NOS) */

	/* alocação de memoria para o vetor de coordenadas */
	coord = (float *)malloc(3*NE*sizeof(float));

	posiciona(&temp, &arq1, "(3i8,6e16.9)", 12);

	for(i=0;i<NE;i++)
	{
		fgets(temp, TAM, arq1);

		CoordT1=9999;		// como o ansys pode não escreve variavel nenhuma quando
		CoordT2=9999;		// o valor desta é zero, precisamos de uma condição para que
		CoordT3=9999;		// a leitura identifique se o que foi lido é ou não uma coord valida

		sscanf( temp, "%d %d %d %f %f %f", &AuxCoord1, &AuxCoord2, &AuxCoord3, &CoordT1, &CoordT2, &CoordT3 );

		if(CoordT1==9999)
			CoordT1 = 0;

		if(CoordT2==9999)
			CoordT2 = 0;

		if(CoordT3==9999)
			CoordT3 = 0;

		coord[i] = CoordT1;
		coord[NE+i] = CoordT2;
		coord[(2*NE)+i] = CoordT3;
	}  // Fim for(i=0;i<NE;i++)

///////////////////////////////////////////////////////////////////////////////////////

/* VETOR "CONECT" (GRAUS DE LIBERDADE DOS NOS) */

	/* alocação de memoria para o vetor de conectividade */

	conect = (int *)malloc(8*M*sizeof(int ));
	cx = (float *)malloc(M*sizeof(float ));
	cy = (float *)malloc(M*sizeof(float ));
	cz = (float *)malloc(M*sizeof(float ));

	posiciona(&temp, &arq1, "(18i7)", 6);

	for(uflag=0;uflag<M;uflag++)
	{
		fgets(temp, TAM, arq1);
		sscanf( temp, "%d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d %d", &b1, &b2, &b3, &ElemAtual, &b4, &b5, &b6, &b7, &b8, &b9, &No1, &No2, &No3, &No4, &No5, &No6, &No7, &No8);

		conect[uflag*8] = No1;
		conect[uflag*8+1] = No2;
		conect[uflag*8+2] = No3;
		conect[uflag*8+3] = No4;
		conect[uflag*8+4] = No5;
		conect[uflag*8+5] = No6;
		conect[uflag*8+6] = No7;
		conect[uflag*8+7] = No8;

                cx[uflag] = (coord[No1-1] + coord[No2-1] + coord[No3-1] + coord[No4-1] + coord[No5-1] + coord[No6-1] + coord[No7-1] + coord[No8-1])/8;
                cy[uflag] = (coord[No1+NE-1] + coord[No2+NE-1] + coord[No3+NE-1] + coord[No4+NE-1] + coord[No5+NE-1] + coord[No6+NE-1] + coord[No7+NE-1] + coord[No8+NE-1])/8;
                cz[uflag] = (coord[No1+(2*NE-1)] + coord[No2+(2*NE-1)] + coord[No3+(2*NE-1)] + coord[No4+(2*NE-1)] + coord[No5+(2*NE-1)] + coord[No6+(2*NE-1)] + coord[No7+(2*NE-1)] + coord[No8+(2*NE-1)])/8;

	} // Fim for(i=0;i<M;i++)

///////////////////////////////////////////////////////////////////////////////////////

/* MODULO DE ELASTICIDIDADE "E" */

	posiciona(&temp, &arq1, "MPDATA,R5.0, 1,EX", 17);
	sscanf( temp, "%s %s %c %s %s %lf", c3, c4, &c5, c6, c7, &E );
	printf("O valor de E e: %f\n", E);

///////////////////////////////////////////////////////////////////////////////////////

/* COEFICIENTE DE POISSON "NI" */

	posiciona(&temp, &arq1, "MPDATA,R5.0, 1,PRXY,", 20);
	sscanf( temp, "%s %s %s %s %lf", g1, g2, g3, g4, &V );
	printf("O valor de V eh: %f\n", V);

///////////////////////////////////////////////////////////////////////////////////////

/* MATRIZ DE RESTRIÇOES "spc" (GRAUS DE LIBERDADE FIXOS) */

	spc12 = (int *)malloc(3*NE*sizeof(int ));
	spc3  = (int *)malloc(3*NE*sizeof(int ));
	spc4  = (int *)malloc(3*NE*sizeof(int ));

	for (uflag=0; uflag<3*NE; uflag++)
	{
		spc12[uflag]=1;
		spc3[uflag]=1;
		spc4[uflag]=1;
	}

	posiciona(&temp, &arq1, "D,", 2);

	k = 0;
	while (k == 0)
	{
		sscanf(temp, "%2s%ld%3s", d1, &d2, d3);
		sscanf(d3, "%c %c %c", &e1, &e2, &e3);

		if (e3 == 'X')
		{
			spc12[3*d2-3]=0;
			spc3[3*d2-3]=0;
			spc4[3*d2-3]=0;
		}

		if (e3 == 'Y')
		{
			spc12[3*d2-2]=0;
			spc3[3*d2-2]=0;
			spc4[3*d2-2]=0;
		}

		if (e3 == 'Z')
		{
			spc12[3*d2-1]=0;
			spc3[3*d2-1]=0;
			spc4[3*d2-1]=0;
		}

		fgets(temp, TAM, arq1);
		k = strncmp( temp, "D,", 2);
	}

/////////////////////////////////////////////////////////////////////////////////////

/* MATRIZ DE FORÇAS "carga14" */

	carga14 = (double *)malloc(3*NE*sizeof(double));

	for (uflag=0; uflag<3*NE; uflag++)
		carga14[uflag]=0;

	k = strncmp( temp, "F,", 2);

	while (k == 0)
	{
		sscanf(temp, "%4s%d%7s%s%f", AuxF1, &d2, AuxF2, AuxF3, &valor_atual );
		sscanf(AuxF2, "%c %c %c %c", &AuxF4, &AuxF5, &direcao_atual, &AuxF6);

		if (direcao_atual == 'X')
		{
			carga14[3*d2-3]=valor_atual;
			spc3[3*d2-3] = 0;
		}

		if (direcao_atual == 'Y')
		{
			carga14[3*d2-2]=valor_atual;
			spc3[3*d2-2] = 0;
		}

		if (direcao_atual == 'Z')
		{
			carga14[3*d2-1]=valor_atual;
			spc3[3*d2-1] = 0;
		}

		fgets(temp, TAM, arq1);
		k = strncmp( temp, "F,", 2);
	}

/////////////////////////////////////////////////////////////////////////////////////

	fclose( arq1 );		//final da leitura de dados do caso 1

/////////////////////////////////////////////////////////////////////////////////////

 
/* Abertura do arquivo .cdb */

	arq1=fopen(entrada2,"r");

//////////////////////////////////////////////////////////////////////////////////////

// Leitura de dados específicos do caso 2:


/* MATRIZ DE FORÇAS "carga23" */

	carga23 = (double *)malloc(3*NE*sizeof(double));

	for (uflag=0; uflag<3*NE; uflag++)
		carga23[uflag]=0;

	posiciona(&temp, &arq1, "F,", 2);

	do
	{
		sscanf(temp, "%4s%d%7s%s%f", AuxF1, &d2, AuxF2, AuxF3, &valor_atual );
		sscanf(AuxF2, "%c %c %c %c", &AuxF4, &AuxF5, &direcao_atual, &AuxF6);

		if (direcao_atual == 'X')
		{
			carga23[3*d2-3]=valor_atual;
			spc4[3*d2-3] = 0;
		}

		if (direcao_atual == 'Y')
		{
			carga23[3*d2-2]=valor_atual;
			spc4[3*d2-2] = 0;
		}

		if (direcao_atual == 'Z')
		{
			carga23[3*d2-1]=valor_atual;
			spc4[3*d2-1] = 0;
		}

		fgets(temp, TAM, arq1);
		k = strncmp( temp, "F,", 2);
	} while (k == 0);

/////////////////////////////////////////////////////////////////////////////////////

	fclose( arq1 );		//final da leitura de dados do caso 2

/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////FIM DA LEITURA DE DADOS//////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////MATRIZES FIXAS///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

/* Vetores "ID" (MATRIZ DE DOF's NAO RESTRITOS ENUMERADOS) */

	ID12 = (int *)malloc(3*NE*sizeof(int));
	ID3 = (int *)malloc(3*NE*sizeof(int));		// Inicialização
	ID4 = (int *)malloc(3*NE*sizeof(int));

	ksize12=0;
	for (i=0; i<3*NE; i++)
	{
		if (spc12[i]==0)
		{
			ID12[i]=0;
		}
		else							// casos 1 e 2
		{
			ksize12++;
			ID12[i]=ksize12;
		}
	}

	ksize3=0;
	for (i=0; i<3*NE; i++)
	{
		if (spc3[i]==0)
		{
			ID3[i]=0;
		}
		else							// caso 3
		{
			ksize3++;
			ID3[i]=ksize3;
		}
	}

	ksize4=0;
	for (i=0; i<3*NE; i++)
	{
		if (spc4[i]==0)
		{
			ID4[i]=0;
		}
		else							// caso 4
		{
			ksize4++;
			ID4[i]=ksize4;
		}
	}

/////////////////////////////////////////////////////////////////////////////////////

/* VETOR DE CARGAS "f1" */

	f1 = (double *)malloc(ksize12*sizeof(double));
	f2 = (double *)malloc(ksize12*sizeof(double));
	f3 = (double *)malloc(ksize3*sizeof(double));
	f4 = (double *)malloc(ksize4*sizeof(double));

	for (uflag=0; uflag<ksize12; uflag++)
	{
		f1[uflag] = 0;
		f2[uflag] = 0;
	}

	for (uflag=0; uflag<ksize3; uflag++)
		f3[uflag] = 0;

	for (uflag=0; uflag<ksize4; uflag++)
		f4[uflag] = 0;

	for (j=0;j<3*NE;j++)
	{
		if (ID12[j]!=0)
		{
			f1[(ID12[j])-1] = carga14[j];
			f2[(ID12[j])-1] = carga23[j];
		}
		if (ID3[j]!=0)
			f3[(ID3[j])-1] = -1*carga23[j];
		if (ID4[j]!=0)
			f4[(ID4[j])-1] = -1*carga14[j];
	}

/////////////////////////////////////////////////////////////////////////////////////

/* MATRIZ "EDOF" (GRAUS DE LIBERDADE) */

	edof = (int *)malloc(24*M*sizeof(int));
	j=0;
	for (i=0; i<8*M; i++)
	{
		edof[j] = 3*conect[i]-2;
		edof[j+1] = 3*conect[i]-1;
		edof[j+2] = 3*conect[i];
		j=j+3;
	}

/////////////////////////////////////////////////////////////////////////////////////

/* Vetores "LM" */

	LM12 = (int *)malloc(24*M*sizeof(int));
	LM3 = (int *)malloc(24*M*sizeof(int));
	LM4 = (int *)malloc(24*M*sizeof(int));

	// Inicialização dos LM
	for (k=0; k<24*M; k++)
	{
		LM12[k]=0;
		LM3[k]=0;
		LM4[k]=0;
	}

	for (i=0; i<M; i++)
		for (j=0; j<=7; j++)
			for (k=0; k<=2; k++)
			{
				LM12[(24*i)+3*j+k] = ID12[k + edof[j*3+(24*i)] - 1];
				 LM3[(24*i)+3*j+k] =  ID3[k + edof[j*3+(24*i)] - 1];
				 LM4[(24*i)+3*j+k] =  ID4[k + edof[j*3+(24*i)] - 1];
			}

	/* Liberação da memória dos vetores usados na leitura do arquivo e geração dos */
	/* dos vetores necessários */
	free(temp);
	free(spc12);
	free(spc3);
	free(spc4);
	free(carga14);
	free(carga23);
	free(ID12);
	free(ID3);
	free(ID4);
	free(edof);

/////////////////////////////////////////////////////////////////////////////////////
///////////////////////////ALOCAÇÃO DE MEMÓRIA///////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////


	xnew = (double *)malloc(M*sizeof(double ));
	xold = (double *)malloc(M*sizeof(double ));
	xupper = (double *)malloc(M*sizeof(double ));
	xlower = (double *)malloc(M*sizeof(double ));
	sinal1 = (double *)malloc(M*sizeof(double ));
	sinal2 = (double *)malloc(M*sizeof(double ));
	sinal3 = (double *)malloc(M*sizeof(double ));
	ml = (double *)malloc(M*sizeof(double ));
	u1 = (double *)malloc(ksize12*sizeof(double));
	u2 = (double *)malloc(ksize12*sizeof(double));
	u3 = (double *)malloc(ksize3*sizeof(double));
	u4 = (double *)malloc(ksize4*sizeof(double));
	difer = (double *)malloc(MAXITER*sizeof(double));
	volume = (double *)malloc(MAXITER*sizeof(double));
	EnMut = (double *)malloc(MAXITER*sizeof(double));
	FM11 = (double *)malloc(MAXITER*sizeof(double));
	FM22 = (double *)malloc(MAXITER*sizeof(double));
	FM33 = (double *)malloc(MAXITER*sizeof(double));
	FM44 = (double *)malloc(MAXITER*sizeof(double));
	objetivo = (double *)malloc(MAXITER*sizeof(double));
	GradEM = (double *)malloc(M*sizeof(double));
	Grad11 = (double *)malloc(M*sizeof(double));
	Grad22 = (double *)malloc(M*sizeof(double));
	Grad33 = (double *)malloc(M*sizeof(double));
	Grad44 = (double *)malloc(M*sizeof(double));
	GradObjetivo = (double *)malloc(M*sizeof(double));
	bf12 = dvector(1,ksize12);
	ubif12= dvector(1,ksize12);
	bf3 = dvector(1,ksize3);
	ubif3= dvector(1,ksize3);
	bf4 = dvector(1,ksize4);
	ubif4= dvector(1,ksize4);
	ija = lvector(1,NMAX);
	sa = dvector(1,NMAX);
	viz12 = imatrix(1,ksize12,1,81);
    viz3 = imatrix(1,ksize3,1,81);
    viz4 = imatrix(1,ksize4,1,81);
    mre = f3tensor(0,(M-1),1,24,1,24);

	// Leitura de valores do usuário
	printf("\n\n");
	printf("Digite o valor das densidades iniciais: ");
	scanf("%f",&x0);
	printf("\nDigite o valor da quantidade de material maxima permitida: ");
	scanf("%f",&Vfrac);

	printf("\nInicializando variáveis\n");
	// inicialização da matriz das densidades
	penal = PEN1;
	for (i=0; i<M; i++)
	{
		sinal1[i] = 1;
		sinal2[i] = 1;
		sinal3[i] = 1;
		ml[i] = 0.04;
		xold[i]=x0;
	}
	//chutes iniciais para a rotina linbcg
	for (uflag=0; uflag<ksize12; uflag++)
	{
		u1[uflag]=0.0;
		u2[uflag]=0.0;
			for(i=1;i<=81;i++)
				viz12[uflag+1][i] = -1;
	}
	for (uflag=0; uflag<ksize3; uflag++)
    {
		u3[uflag]=0.0;
			for(i=1;i<=81;i++)
				viz3[uflag+1][i] = -1;
    }
    for (uflag=0; uflag<ksize4; uflag++)
    {
		u4[uflag]=0.0;
			for(i=1;i<=81;i++)
				viz4[uflag+1][i] = -1;
	}

    for (i=0;i<MAXITER;i++)
	{
		difer[i]=1;
		EnMut[i]=0;
		FM11[i]=0;
		FM22[i]=0;
		FM33[i]=0;
		FM44[i]=0;
	}
	volume[0] = M*x0;
	printf("OK!!!!!\n");

        // monta vizinhos
        printf("Montando vizinhos...\n");
        for(k=0;k<M;k++)
        {
                for(i=0;i<24;i++)
                {
                        uflag = LM12[24*k + i];
                        if(uflag!=0)            // if LM12 não restrito
                        {
                                for(j=0;j<24;j++)
                                {
                                        d2 = LM12[24*k + j];
                                        if(d2 != 0 && d2!= uflag)             // if vizinho não restrito
                                        {
                                                for(it=1;viz12[uflag][it]<d2 && viz12[uflag][it]!=-1;it++);
                                                if(viz12[uflag][it] > d2)
                                                {
                                                        for(iter=80;iter>it;iter--)
                                                                viz12[uflag][iter] = viz12[uflag][iter-1];
                                                        viz12[uflag][it] = -1;
                                                }
                                                if(viz12[uflag][it] == -1)
                                                        viz12[uflag][it] = d2;
                                        } // fim if vizinho não restrito
                                } // fim for j LM12
                        } // fim if LM12 não restrito

                        uflag = LM3[24*k + i];
                        if(uflag!=0)            // if LM3 não restrito
                        {
                                for(j=0;j<24;j++)
                                {
                                        d2 = LM3[24*k + j];
                                        if(d2 != 0 && d2!= uflag)             // if vizinho não restrito
                                        {
                                                for(it=1;viz3[uflag][it]<d2 && viz3[uflag][it]!=-1;it++);
                                                if(viz3[uflag][it] > d2)
                                                {
                                                        for(iter=80;iter>it;iter--)
                                                                viz3[uflag][iter] = viz3[uflag][iter-1];
                                                        viz3[uflag][it] = -1;
                                                }
                                                if(viz3[uflag][it] == -1)
                                                        viz3[uflag][it] = d2;
                                        } // fim if vizinho não restrito
                                } // fim for j LM3
                        } // fim if LM3 não restrito

                        uflag = LM4[24*k + i];
                        if(uflag!=0)            // if LM4 não restrito
                        {
                                for(j=0;j<24;j++)
                                {
                                        d2 = LM4[24*k + j];
                                        if(d2 != 0 && d2!= uflag)             // if vizinho não restrito
                                        {
                                                for(it=1;viz4[uflag][it]<d2 && viz4[uflag][it]!=-1;it++);
                                                if(viz4[uflag][it] > d2)
                                                {
                                                        for(iter=80;iter>it;iter--)
                                                                viz4[uflag][iter] = viz4[uflag][iter-1];
                                                        viz4[uflag][it] = -1;
                                                }
                                                if(viz4[uflag][it] == -1)
                                                        viz4[uflag][it] = d2;
                                        } // fim if vizinho não restrito
                                } // fim for j LM4
                        } // fim if LM4 não restrito
                } // fim for i
        } // fim for monta vizinhos
        printf("OK!!!!!\n");

	 // Constrói as matrizes de rigidez para todos elementos
        printf("Montando matrizes de rigidez locais...\n");
	for(i=0;i<M;i++)
		SKE(i,conect,coord,E,V,mre[i],NE);
        printf("OK!\n\n");

//////////////////////////Condições iniciais//////////////////////////////
	printf("Calculando as condições iniciais da estrutura:\n\n");
	it=0;
	// matriz de rigidez global para carregamentos 1 e 2
	MontaGlobal(ksize12,M,NE,LM12,viz12,xold,penal,mre);

	for (uflag=1;uflag<=ksize12;uflag++)
	{
		bf12[uflag]=f1[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
		ubif12[uflag]=u1[uflag-1];		// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
	}
	linbcg(ksize12, bf12, ubif12, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u1=f1
	for (uflag=0; uflag<ksize12; uflag++)
		u1[uflag] = ubif12[uflag+1];		// retorna resultado para matriz verdadeira

	for (uflag=1;uflag<=ksize12;uflag++)
	{
		bf12[uflag]=f2[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
		ubif12[uflag]=u2[uflag-1];		// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
	}
	linbcg(ksize12, bf12, ubif12, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u2=f2
	for (uflag=0; uflag<ksize12; uflag++)
		u2[uflag] = ubif12[uflag+1];		// retorna resultado para matriz verdadeira

	for (uflag=0;uflag<ksize12;uflag++)
		EnMut[it] += u1[uflag]*f2[uflag];		// calcula energia mútua
	printf("Energia Mutua = %.10f\n", (EnMut[it]));

	for (uflag=0;uflag<ksize12;uflag++)
		FM11[it] += -1*u1[uflag]*f1[uflag];		// calcula flexibilidade média 11
	printf("FM11 = %.10f\n", (FM11[it]));

	for (uflag=0;uflag<ksize12;uflag++)
		FM22[it] += -1*u2[uflag]*f2[uflag];		// calcula flexibilidade média 22
	printf("FM22 = %.10f\n", (FM22[it]));

	// matriz de rigidez global para carregamento 3
	MontaGlobal(ksize3,M,NE,LM3,viz3,xold,penal,mre);
	for (uflag=1;uflag<=ksize3;uflag++)
	{
		bf3[uflag]=f3[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
		ubif3[uflag]=u3[uflag-1];			// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
	}
	linbcg(ksize3, bf3, ubif3, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u3=f3
	for (uflag=0; uflag<ksize3; uflag++)
		u3[uflag] = ubif3[uflag+1];		// retorna resultado para matriz verdadeira

	for (uflag=0;uflag<ksize3;uflag++)
		FM33[it] += u3[uflag]*f3[uflag];		// calcula flexibilidade média 33
        printf("FM33 = %.10f\n", (FM33[it]));

	// matriz de rigidez global para carregamento 4
	MontaGlobal(ksize4,M,NE,LM4,viz4,xold,penal,mre);
	for (uflag=1;uflag<=ksize4;uflag++)
	{
		bf4[uflag]=f4[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
		ubif4[uflag]=u4[uflag-1];			// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
	}
	linbcg(ksize4, bf4, ubif4, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u4=f4
	for (uflag=0; uflag<ksize4; uflag++)
		u4[uflag] = ubif4[uflag+1];		// retorna resultado para matriz verdadeira

	for (uflag=0;uflag<ksize4;uflag++)
		FM44[it] += u4[uflag]*f4[uflag];		// calcula flexibilidade média 44
	printf("FM44 = %.10f\n", (FM44[it]));

	objetivo[it] = EnMut[it]/(w1*FM11[it] + w2*FM22[it] + w3*FM33[it] + w4*FM44[it]);
	objetivold = objetivo[it];

	printf("Condições iniciais: obj: %f\t vol: %f\n\n\n", objetivold, volume[it]);
//////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////LOOP PRINCIPAL/////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
	opcao = 1;
	op2 = 1;
	b1 = 0;
	b2 = MAXITER;
	b3 = MAXITER;
	while(opcao)
	{
		if(b1 == 2)
		{
				//fornecimento de dados para arquivo em ANSYS
			printf("Escrevendo arquivo para AnSys...\n");
			switch(b3)
			{
				case 0:
					arq3=fopen(A0,"wb");
					break;
				case 1:
					arq3=fopen(A1,"wb");
					break;
				case 2:
					arq3=fopen(A2,"wb");
					break;
				case 3:
					arq3=fopen(A3,"wb");
					break;
				case 4:
					arq3=fopen(A4,"wb");
					break;
				case 5:
					arq3=fopen(A5,"wb");
					break;
				case 6:
					arq3=fopen(A6,"wb");
					break;
				case 7:
					arq3=fopen(A7,"wb");
					break;
				case 8:
					arq3=fopen(A8,"wb");
					break;
				case 9:
					arq3=fopen(A9,"wb");
					break;
			}

			fprintf(arq3, "/SHOW\n\n");
			fprintf(arq3, "/UNITS,CGS\n");
			fprintf(arq3, "/TITLE,Modelo Otimizado\n");
			fprintf(arq3, "/PREP7\n");
			fprintf(arq3, "ET,1,SOLID45\n\n");
			// materiais
			fprintf(arq3, "MPTEMP,,,,,,,, \n");
			fprintf(arq3, "MPTEMP,1,0\n");
			fprintf(arq3, "MPDATA,EX,1,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,1,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,2,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,2,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,3,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,3,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,4,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,4,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,5,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,5,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,6,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,6,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,7,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,7,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,8,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,8,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,9,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,9,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,10,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,10,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,11,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,11,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,12,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,12,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,13,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,13,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,14,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,14,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,15,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,15,,%f\n",V);
			fprintf(arq3, "MPDATA,EX,16,,%f\n",E);
			fprintf(arq3, "MPDATA,PRXY,16,,%f\n",V);

			// Coordenadas dos nós
			for (i=0; i < NE; i++)
				fprintf(arq3, "N,%d,%f,%f,%f\n", i+1, coord[i], coord[NE+i], coord[2*NE+i]);
			fprintf(arq3, "\n");

			// Elementos
			fprintf(arq3, "REAL,1\n\n");
			fprintf(arq3, "\n/CMAP,FILE,CMAP\n");
			fprintf(arq3, "/COLOR,OUTL,15\n");
			for (i=0; i<M; i++)
			{
				fprintf(arq3, "E,%d,%d,%d,%d,%d,%d,%d,%d\n", conect[i*8], conect[i*8+1], conect[i*8+2], conect[i*8+3], conect[i*8+4], conect[i*8+5], conect[i*8+6], conect[i*8+7]);

				if (xold[i] <= 0.06)
				{
					fprintf(arq3, "EMODIF,%d,MAT,16\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,0,%d\n", i+1);
				}
				else if (xold[i] <= 0.12)
				{
					fprintf(arq3, "EMODIF,%d,MAT,15\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,1,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.18)
				{
					fprintf(arq3, "EMODIF,%d,MAT,14\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,2,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.24)
				{
					fprintf(arq3, "EMODIF,%d,MAT,13\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,3,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.30)
				{
					fprintf(arq3, "EMODIF,%d,MAT,12\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,4,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.36)
				{
					fprintf(arq3, "EMODIF,%d,MAT,11\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,5,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.42)
				{
					fprintf(arq3, "EMODIF,%d,MAT,10\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,6,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.48)
				{
					fprintf(arq3, "EMODIF,%d,MAT,9\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,7,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.54)
				{
					fprintf(arq3, "EMODIF,%d,MAT,8\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,8,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.60)
				{
					fprintf(arq3, "EMODIF,%d,MAT,7\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,9,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.66)
				{
					fprintf(arq3, "EMODIF,%d,MAT,6\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,10,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.72)
				{
					fprintf(arq3, "EMODIF,%d,MAT,5\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,11,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.78)
				{
					fprintf(arq3, "EMODIF,%d,MAT,4\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,12,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.84)
				{
					fprintf(arq3, "EMODIF,%d,MAT,3\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,13,%d\n", i+1);      // densidades dos elementos
				}
				else if (xold[i] <= 0.90)
				{
					fprintf(arq3, "EMODIF,%d,MAT,2\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,14,%d\n", i+1);
				}
				else
				{
					fprintf(arq3, "EMODIF,%d,MAT,1\n", i+1);
					fprintf(arq3, "/COLOR,ELEM,15,%d\n", i+1);
				}
			}  // fim for i

			fprintf(arq3, "\nFINISH");
			fclose(arq3);
			printf("OK\n\n");
			// fim do arquivo do AnSys
			b3++;
		} // fim if b1 == 2

		if( difer[it]<ITERTOL && b1 == 1 && it>b3)
		{
			printf("\nDeseja continuar a busca?(s = 1, n = 0): ");
			scanf("%d", &op2);
			if(op2 == 0)
			{
				b1 = 2;                 // flag, deixa de usar o filtro
				b2 = it + 11;            // mais doze iterações com o filtro desligado
				b3 = 0;
				printf("\n\nDesligando o filtro espacial.\nPreparando iterações finais\n\n");
			} // fim if
		} // fim if
		else if(penal == PEN1 && it>20 && difer[it]<CONT)
		{
			penal = PEN2;	// critério da continualidade
			b1 = 1;			// flag, liga o filtro
			b3 = it+15;
			printf("\nLigando o filtro espacial.\n\n");
		}

//////////////////////////////////////////////////////////////////////////////////////
		//////////////GRADIENTE DA FUNÇÃO OBJETIVO///////////////////////////
//////////////////////////////////////////////////////////////////////////////////////
		for (i=0;i<M;i++)
		{
			/* inicialização dos vetores de gradientes */
			GradEM[i]=0;
			Grad11[i]=0;
			Grad22[i]=0;
			Grad33[i]=0;
			Grad44[i]=0;

			//extrai de u1 e u2 os vetores de deslocamentos de cada elemento para carregamentos 1 e 2
			for (j=0;j<24;j++)
			{
				if (LM12[i*24+j]!=0)
				{
					uea[j]=u1[LM12[i*24+j]-1];
					ueb[j]=u2[LM12[i*24+j]-1];
				}
				else
				{
					uea[j]=0;
					ueb[j]=0;
				}
			}

			// calcula o esforço em cada elemento para energia mutua
			for (j=0; j<24; j++)
			{
				tempmat[j]=0;
				for (k=0; k<24; k++)
					tempmat[j] += uea[k]*mre[i][j+1][k+1];
			}
			// Calcula a derivada da energia mutua pela densidade para o elemento i
			for (j=0; j<24; j++)
				GradEM[i] += -1*penal*pow(xold[i],(penal-1))*tempmat[j]*ueb[j];

			// calcula o esforço em cada elemento para flex med 11
			for (j=0; j<24; j++)
			{
				tempmat[j]=0;
				for (k=0; k<24; k++)
					tempmat[j] += uea[k]*mre[i][j+1][k+1];
			}
			// Calcula a derivada da FM11 pela densidade para o elemento i
			for (j=0; j<24; j++)
				Grad11[i] += -1*penal*pow(xold[i],(penal-1))*tempmat[j]*uea[j];

			// calcula o esforço em cada elemento para flex med 22
			for (j=0; j<24; j++)
			{
				tempmat[j]=0;
				for (k=0; k<24; k++)
					tempmat[j] += ueb[k]*mre[i][j+1][k+1];
			}
			// Calcula a derivada da FM22 pela densidade para o elemento i
			for (j=0; j<24; j++)
				Grad22[i] += -1*penal*pow(xold[i],(penal-1))*tempmat[j]*ueb[j];

			//extrai de u3 o vetor de deslocamentos de cada elemento para carregamento 3
			for (j=0;j<24;j++)
			{
				if (LM3[i*24+j]!=0)
					uea[j] = u3[ LM3[i*24+j] - 1];
				else
					uea[j] = 0;
			}

			// calcula o esforço em cada elemento para flexibilidade média 33
			for (j=0; j<24; j++)
			{
				tempmat[j]=0;
				for (k=0; k<24; k++)
					tempmat[j] += uea[k]*mre[i][j+1][k+1];
			}
			// Calcula a derivada da FM44 pela densidade para o elemento i
			for (j=0; j<24; j++)
				Grad33[i] += -1*penal*pow(xold[i],(penal-1))*tempmat[j]*uea[j];

			//extrai de u4 o vetor de deslocamentos de cada elemento para carregamento 4
			for (j=0;j<24;j++)
			{
				if (LM4[i*24+j]!=0)
					uea[j] = u4[ LM4[i*24+j] - 1];
				else
					uea[j] = 0;
			}

			// calcula o esforço em cada elemento para flexibilidade média 44
			for (j=0; j<24; j++)
			{
				tempmat[j]=0;
				for (k=0; k<24; k++)
					tempmat[j] += uea[k]*mre[i][j+1][k+1];
			}
			// Calcula a derivada da FM44 pela densidade para o elemento i
			for (j=0; j<24; j++)
				Grad44[i] += -1*penal*pow(xold[i],(penal-1))*tempmat[j]*uea[j];

			GradObjetivo[i] = GradEM[i]/(w1*FM11[it] + w2*FM22[it] + w3*FM33[it] + w4*FM44[it]);
			GradObjetivo[i] -= EnMut[it]*(w1*Grad11[i] + w2*Grad22[i] + w3*Grad33[i] + w4*Grad44[i])/((w1*FM11[it] + w2*FM22[it] + w3*FM33[it] + w4*FM44[it])*(w1*FM11[it] + w2*FM22[it] + w3*FM33[it] + w4*FM44[it]));
		}  // fim do loop de cálculo do gradiente

		it++;

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////Atualização das variáveis////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

		limites(M,sinal1,sinal2,sinal3,ml,xold,xupper,xlower);
		if(b1 == 1) filtro(M,cx,cy,cz,xupper,xlower,RMAX);

		/* Critério da optimalidade */
		volume[it] = OC(xold,xnew,GradObjetivo,xlower,xupper,Vfrac,M);
		for(uflag = 0; uflag < M; uflag++)
		{
			sinal3[uflag] = sinal2[uflag];
			sinal2[uflag] = sinal1[uflag];
			sinal1[uflag] = xnew[uflag] - xold[uflag];
			xold[uflag] = xnew[uflag];
		}

//////////////////////////////////////////////////////////////////////////////////////
/////////////////////// CÁLCULO DA FUNÇÃO OBJETIVO ///////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////

		// matriz de rigidez global para carregamentos 1 e 2
		MontaGlobal(ksize12,M,NE,LM12,viz12,xold,penal,mre);

		for (uflag=1;uflag<=ksize12;uflag++)
		{
			bf12[uflag]=f1[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
			ubif12[uflag]=u1[uflag-1];		// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
		}
		linbcg(ksize12, bf12, ubif12, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u1=f1
		for (uflag=0; uflag<ksize12; uflag++)
			u1[uflag] = ubif12[uflag+1];		// retorna resultado para matriz verdadeira

		for (uflag=1;uflag<=ksize12;uflag++)
		{
			bf12[uflag]=f2[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
			ubif12[uflag]=u2[uflag-1];		// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
		}
		linbcg(ksize12, bf12, ubif12, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u2=f2
		for (uflag=0; uflag<ksize12; uflag++)
			u2[uflag] = ubif12[uflag+1];		// retorna resultado para matriz verdadeira

		for (uflag=0;uflag<ksize12;uflag++)
			EnMut[it] += u1[uflag]*f2[uflag];		// calcula energia mútua
		printf("Energia Mutua = %.10f\n", (EnMut[it]));

		for (uflag=0;uflag<ksize12;uflag++)
			FM11[it] += -1*u1[uflag]*f1[uflag];		// calcula flexibilidade média 11
		printf("FM11 = %.10f\n", (FM11[it]));

		for (uflag=0;uflag<ksize12;uflag++)
			FM22[it] += -1*u2[uflag]*f2[uflag];		// calcula flexibilidade média 22
		printf("FM22 = %.10f\n", (FM22[it]));

		// matriz de rigidez global para carregamento 3
		MontaGlobal(ksize3,M,NE,LM3,viz3,xold,penal,mre);
		for (uflag=1;uflag<=ksize3;uflag++)
		{
			bf3[uflag]=f3[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
			ubif3[uflag]=u3[uflag-1];			// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
		}
		linbcg(ksize3, bf3, ubif3, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u3=f3
		for (uflag=0; uflag<ksize3; uflag++)
			u3[uflag] = ubif3[uflag+1];		// retorna resultado para matriz verdadeira

		for (uflag=0;uflag<ksize3;uflag++)
			FM33[it] += u3[uflag]*f3[uflag];		// calcula flexibilidade média 33
        printf("FM33 = %.10f\n", (FM33[it]));

		// matriz de rigidez global para carregamento 4
		MontaGlobal(ksize4,M,NE,LM4,viz4,xold,penal,mre);
		for (uflag=1;uflag<=ksize4;uflag++)
		{
			bf4[uflag]=f4[uflag-1];			// Prepara vetores de forças e deslocamentos p/ linbcg
			ubif4[uflag]=u4[uflag-1];			// bf e ubif vão de 1 a ksize!!! (Não de 0 a ksize-1)
		}
		linbcg(ksize4, bf4, ubif4, TOL, ITMAXLBCG, &iter, &err);  // Resolve sistema K*u4=f4
		for (uflag=0; uflag<ksize4; uflag++)
			u4[uflag] = ubif4[uflag+1];		// retorna resultado para matriz verdadeira

		for (uflag=0;uflag<ksize4;uflag++)
			FM44[it] += u4[uflag]*f4[uflag];		// calcula flexibilidade média 44
		printf("FM44 = %.10f\n", (FM44[it]));

		objetivo[it] = EnMut[it]/(w1*FM11[it] + w2*FM22[it] + w3*FM33[it] + w4*FM44[it]);

		//Verificaçao da condiçoes de loop
		difer[it] = fabs((objetivo[it]-objetivold)/objetivold);
		objetivold = objetivo[it];

		printf("iteracao %d\t obj: %f\t vol: %f\t difer: %f\n", it, objetivold, volume[it], difer[it]);
		printf("Fim da %da iteração.\n\n\n", it);

		if (it==MAXITER)
		{
			opcao = 0;
			printf("\n\nAtingido o número máximo de iterações - abortando o programa.\n\n");
		}
		else if (it>=b2)
		{
			printf("\nConvergência atingida, iterações pós-filtro realizadas. Finalizando o programa.\n\n");
			opcao = 0;
		}
//////////////////////////////////////////////////////////////////////////////////
	} /* fim do loop */

	free_dvector(sa,1,NMAX);
	free_lvector(ija,1,NMAX);
    free_f3tensor(mre,0,(M-1),1,24,1,24);
    free_dvector(ubif4,1,ksize4);
    free_dvector(bf4,1,ksize4);
    free_dvector(ubif3,1,ksize3);
    free_dvector(bf3,1,ksize3);
    free_dvector(ubif12,1,ksize12);
    free_dvector(bf12,1,ksize12);
    free(GradObjetivo);
    free(Grad44);
	free(Grad33);
    free(Grad22);
    free(Grad11);
    free(GradEM);
    free(u4);
    free(u3);
    free(u2);
    free(u1);
	free(xupper);
	free(xlower);
	free(xnew);

/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////CRIAÇAO DO ARQUIVO MATLAB////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

	printf("Escrevendo arquivo de MatLab\n");
	//Abre o arquivo em MATLAB
	arquivo = fopen(MATLAB,"w");

	//funçao objetivo
	fprintf(arquivo, "obj = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",objetivo[i]);
        fprintf(arquivo, "];\n");

	//volume
	fprintf(arquivo, "vol = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%f ",volume[i]);
        fprintf(arquivo, "];\n");

	// Energia Mútua
	fprintf(arquivo, "EM = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",EnMut[i]);
        fprintf(arquivo, "];\n");

	// Flexibilidade Média 11
	fprintf(arquivo, "FM11 = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",FM11[i]);
        fprintf(arquivo, "];\n");
	// Flexibilidade Média 22
	fprintf(arquivo, "FM22 = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",FM22[i]);
        fprintf(arquivo, "];\n");
	// Flexibilidade Média 33
	fprintf(arquivo, "FM33 = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",FM33[i]);
        fprintf(arquivo, "];\n");
	// Flexibilidade Média 44
	fprintf(arquivo, "FM44 = [");
	for (i=0; i<it; i++)
		fprintf(arquivo,"%.10f ",FM44[i]);
        fprintf(arquivo, "];\n");
		// densidades (só para informação)
	fprintf(arquivo, "densid = [");
	for (i=0; i<M; i++)
		fprintf(arquivo,"%.3f ",xold[i]);
        fprintf(arquivo, "];\n\n");
	//numero de iterações
	fprintf(arquivo, "it = %d;\n\n", it);

	//plotagem dos graficos
	fprintf(arquivo, "figure;\n");
	// Objetivo
	fprintf(arquivo, "subplot(2,2,1);\n");
	fprintf(arquivo, "plot(obj);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Função Objetivo');\n");
	// Volume
	fprintf(arquivo, "subplot(2,2,2);\n");
	fprintf(arquivo, "plot(vol);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Volume total');\n");
	// Energia Mútua
	fprintf(arquivo, "subplot(2,2,3);\n");
	fprintf(arquivo, "plot(EM);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Energia Mútua');\n");

	fprintf(arquivo, "\nfigure;\n");
	// Flexibilidade Média 11
	fprintf(arquivo, "subplot(2,2,1);\n");
	fprintf(arquivo, "plot(FM11);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Flexibilidade Média 11');\n");
	// Flexibilidade Média 2
	fprintf(arquivo, "subplot(2,2,2);\n");
	fprintf(arquivo, "plot(FM22);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Flexibilidade Média 22');\n");
	// Flexibilidade Média 3
	fprintf(arquivo, "subplot(2,2,3);\n");
	fprintf(arquivo, "plot(FM33);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Flexibilidade Média 33');\n");
	// Flexibilidade Média 4
	fprintf(arquivo, "subplot(2,2,4);\n");
	fprintf(arquivo, "plot(FM44);\n");
	fprintf(arquivo, "xlabel('Iteração');\n");
	fprintf(arquivo, "ylabel('Flexibilidade Média 44');\n");

	fclose(arquivo);
	printf("OK\n\n");
	//fim da criação do arquivo de MatLab

///////////////////////////////////////////////////////////////////////////////////////////////

	//fornecimento de dados para arquivo em ANSYS
	printf("Escrevendo arquivo para AnSys com a estrutura final...\n\n");
	arq3=fopen(ANSYS,"wb");

	fprintf(arq3, "/SHOW\n\n");
	fprintf(arq3, "/UNITS,CGS\n");
	fprintf(arq3, "/TITLE,Modelo Otimizado\n");
	fprintf(arq3, "/PREP7\n");
	fprintf(arq3, "ET,1,SOLID45\n\n");
	// materiais
	fprintf(arq3, "MPTEMP,,,,,,,, \n");
	fprintf(arq3, "MPTEMP,1,0\n");
	fprintf(arq3, "MPDATA,EX,1,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,1,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,2,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,2,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,3,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,3,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,4,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,4,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,5,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,5,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,6,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,6,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,7,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,7,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,8,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,8,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,9,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,9,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,10,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,10,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,11,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,11,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,12,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,12,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,13,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,13,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,14,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,14,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,15,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,15,,%f\n\n",V);
	fprintf(arq3, "MPDATA,EX,16,,%f\n",E);
	fprintf(arq3, "MPDATA,PRXY,16,,%f\n\n",V);

	// Coordenadas dos nós
	for (i=0; i < NE; i++)
		fprintf(arq3, "N,%d,%f,%f,%f\n", i+1, coord[i], coord[NE+i], coord[2*NE+i]);
	fprintf(arq3, "\n");

	// Elementos
	fprintf(arq3, "REAL,1\n\n");
	fprintf(arq3, "\n/CMAP,FILE,CMAP\n");
	fprintf(arq3, "/COLOR,OUTL,15\n");
	for (i=0; i<M; i++)
	{

		fprintf(arq3, "E,%d,%d,%d,%d,%d,%d,%d,%d\n", conect[i*8], conect[i*8+1], conect[i*8+2], conect[i*8+3], conect[i*8+4], conect[i*8+5], conect[i*8+6], conect[i*8+7]);

		if (xold[i] <= 0.06)
		{
			fprintf(arq3, "EMODIF,%d,MAT,16\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,0,%d\n", i+1);
		}
		else if (xold[i] <= 0.12)
		{
			fprintf(arq3, "EMODIF,%d,MAT,15\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,1,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.18)
		{
			fprintf(arq3, "EMODIF,%d,MAT,14\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,2,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.24)
		{
			fprintf(arq3, "EMODIF,%d,MAT,13\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,3,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.30)
		{
			fprintf(arq3, "EMODIF,%d,MAT,12\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,4,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.36)
		{
			fprintf(arq3, "EMODIF,%d,MAT,11\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,5,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.42)
		{
			fprintf(arq3, "EMODIF,%d,MAT,10\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,6,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.48)
		{
			fprintf(arq3, "EMODIF,%d,MAT,9\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,7,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.54)
		{
			fprintf(arq3, "EMODIF,%d,MAT,8\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,8,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.60)
		{
			fprintf(arq3, "EMODIF,%d,MAT,7\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,9,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.66)
		{
			fprintf(arq3, "EMODIF,%d,MAT,6\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,10,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.72)
		{
			fprintf(arq3, "EMODIF,%d,MAT,5\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,11,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.78)
		{
			fprintf(arq3, "EMODIF,%d,MAT,4\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,12,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.84)
		{
			fprintf(arq3, "EMODIF,%d,MAT,3\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,13,%d\n", i+1);      // densidades dos elementos
		}
		else if (xold[i] <= 0.90)
		{
			fprintf(arq3, "EMODIF,%d,MAT,2\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,14,%d\n", i+1);
		}
		else
		{
			fprintf(arq3, "EMODIF,%d,MAT,1\n", i+1);
			fprintf(arq3, "/COLOR,ELEM,15,%d\n", i+1);
		}
	}  // fim for i

	fprintf(arq3, "\nFINISH");
	fclose(arq3);
	printf("OK\n\n");
	// fim do arquivo do AnSys

    free(objetivo);
    free(FM44);
    free(FM33);
    free(FM22);
    free(FM11);
    free(EnMut);
    free(volume);
    free(difer);
    free_imatrix(viz4,1,ksize4,1,81);
    free_imatrix(viz3,1,ksize3,1,81);
	free_imatrix(viz12,1,ksize12,1,81);
	free(ml);
	free(sinal3);
	free(sinal2);
	free(sinal1);
	free(xold);

} // Fim main


void limites(int M, double *sign1, double *sign2, double *sign3, double *ml, double *x, double *xupper, double *xlower)
{
	int i;

	for (i=0; i<M; i++)
	{
		if (sign1[i] > 0 && sign2[i] <= 0 && sign3[i] >= 0)
			ml[i] = ml[i] * minf;
		else if (sign1[i] < 0 && sign2[i] >= 0 && sign3[i] <= 0)
			ml[i] = ml[i] * minf;
		else
			ml[i] = ml[i] * msup;

		if (ml[i] > mlupper) ml[i] = mlupper;
		else if (ml[i] < mllower) ml[i] = mllower;

		xlower[i] = x[i] * (1 - ml[i]);
		xupper[i] = x[i] * (1 + ml[i]);

		if (xlower[i] < EPS) xlower[i] = EPS;
		if (xupper[i] > 1.0) xupper[i] = 1.0;
	} // fim for i=1:M
} // fim filtro


void filtro(int M, float *cx, float *cy, float *cz, double *bu, double *bl, double radius)
{
	int i, j, k;
	double fac, sum, somatu, somatl, r;
	double *flow, *fup;

	flow = (double *)malloc(M*sizeof(double ));
	fup = (double *)malloc(M*sizeof(double ));

	for(i=0; i<M; i++)	// calcula filtro
	{
		sum = 0;
		somatu = 0;
		somatl = 0;

		for(j = 0; j < M; j++)
		{
			if(i != j)
			{
				r = distancia(cx[i],cy[i],cz[i],cx[j],cy[j],cz[j]);
				if(r < radius)
				{
					fac = (radius - r)/radius;

					sum += fac;
					somatu += (bu[j] * fac);
					somatl += (bl[j] * fac);
				} // fim if r< RMAX
			}// fim fi i!=j
		} // fim for j

		fup[i] = (bu[i] + somatu) / (1 + sum);
		flow[i] = (bl[i] + somatl) / (1 + sum);
	} // fim for calcula filtro

	for(i=0;i<M;i++) // atualiza limites
	{
		if(flow[i]>=0.0) bl[i] = flow[i]; else bl[i] = 0.0;
		if(fup[i]<=1.0) bu[i] = fup[i]; else bu[i] = 1.0;
	} // fim for atualiza limites

	free(flow);
	free(fup);
}


void posiciona(char **temp, FILE **arq1, char *desejado, int comprimento)
{
	int compara;

	compara = -1;
	while (compara!=0)
	{
		fgets(*temp,TAM,*arq1);
		compara = strncmp(*temp, desejado, comprimento);
	}  // fim while de posicionamento

} // Fim posiciona


void insere(int i, int j, double v)
{
	unsigned int c1;

	if(i == j) sa[i] += v;    // fim if diagonal
	else
	{
		for(c1=ija[i];ija[c1]<j;c1++);  // posiciona
		sa[c1] += v;
	}
} // fim insere


void Prepara(long ksize, int **vizinhos)
{
        long dof,i,n;

        n = ksize+2;
        for(dof=1;dof<=ksize;dof++)
        {
                ija[dof] = n;
                sa[dof] = 0.0;
                for(i=1;vizinhos[dof][i]!=-1;i++)
                {
                        sa[n] = 0.0;
                        ija[n] = vizinhos[dof][i];
                        n++;
                }
        } // fim for dof
        ija[ksize+1] = n;
} // fim PreparaInd


void MontaGlobal(long ksize,int M,int N,int *LM,int **vizinhos,double *x,double p,float ***elementos)
{
	unsigned long i,j,k;
	double valor;
        int *dofs;

        dofs = ivector(1,24);

	//zera a matriz de rigidez global
        Prepara(ksize,vizinhos);

        for(k=0;k<M;k++) // para cada elemento
        {
                for(i=0;i<24;i++)
                        dofs[i+1] = LM[24*k+i];
                for(i=1;i<=24;i++)
                        if( dofs[i] != 0 )
                                for(j=1;j<=24;j++)
                                        if( dofs[j] != 0 )
                                        {
                                                valor = pow(x[k],p)*elementos[k][i][j];
                                                insere(dofs[i],dofs[j],valor);
                                        } // fim if ambos os dofs não restritos
        } // fim for cada elemento

        free_ivector(dofs,1,24);
} // Fim MontaGlobal


float distancia(float x1, float y1, float z1, float x2, float y2, float z2)
{
        float dist;
        dist = (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) + (z2-z1)*(z2-z1);
        dist = sqrt(dist);
        return(dist);
} // fim distancia


double OC(double *x,double *xnew,double *grad,double *xlow,double *xup,float v,int n)
{
	int i;
	double volume;
	double b, l1, l2, lm;

	l1 = 0.0;
	l2 = 1.0e5;
	while( ( ((l2-l1)/(l2+l1))>EPS ) && (l2>EPS) )
	{
		volume = 0.0;
		lm = (l1+l2)/2;
		for(i=0;i<n;i++)
		{
			b = grad[i]/lm;
			if(b<0.0) b = 0.0;
			else b = pow(b,0.3);		// b = min(x*b,0);
			b = x[i]*b;

			if(b>xup[i]) b = xup[i];
			else if(b<xlow[i]) b = xlow[i];		// xlow < b < xup

			if(b>0.0) xnew[i] = b;		// xnew = max(b, 0);
			else xnew[i] = 0.0;

			volume += xnew[i];
		} // fim for cada elemento

		if( volume>(v*n) ) l1 = lm;
		else l2 = lm;
	} // fim while principal

	return(volume);
} // fim OC


void SKE(int el, int *conect, float *coord,double E,double v, float **ske, int NE)
{
	float *xe, *ye, *ze;
	float *N0,J0,*dNdx0,*dNdy0,*dNdz0,*N,J,*dNdx,*dNdy,*dNdz;
	float **D,a,*rgau,*sgau,*tgau,s,r,t,**B,**B2,*rgau2,*sgau2,*tgau2;
	int i,j,k;


	J = 0;
        dNdx = vector(1,8);
	dNdy = vector(1,8);
	dNdz = vector(1,8);
	N = vector(1,8);
	dNdx0 = vector(1,8);
	dNdy0 = vector(1,8);
	dNdz0 = vector(1,8);
	N0 = vector(1,8);
	xe=vector(1,8);
	ye=vector(1,8);
	ze=vector(1,8);

	/* D matrix (isotropic elasticity) */
	D=matrix(1,6,1,6);
	rgau=vector(1,8);
	sgau=vector(1,8);
	tgau=vector(1,8);
	rgau2=vector(1,2);
	sgau2=vector(1,2);
	tgau2=vector(1,2);
	B=matrix(1,6,1,24);
	B2=matrix(1,24,1,6);

	for(k=1;k<=8;k++)
	{
		xe[k] = coord[conect[(k-1)+el*8]-1];
		ye[k] = coord[conect[(k-1)+el*8]+NE-1];
		ze[k] = coord[conect[(k-1)+el*8]+2*NE-1];
	}

	for(i=1;i<=6;i++)
		for(j=1;j<=6;j++)
			D[i][j] = 0;

	D[1][1] = E*(1-v)/((1-2*v)*(1+v));
	D[1][2] = D[1][1]*v/(1-v);
	D[1][3] = D[1][2];
	D[2][1] = D[1][2];
	D[2][2] = D[1][1];
	D[2][3] = D[1][2];
	D[3][1] = D[1][3];
	D[3][2] = D[2][3];
	D[3][3] = D[1][1];
	D[4][4] = E/(2*(1+v));
	D[5][5] = E/(2*(1+v));
	D[6][6] = E/(2*(1+v));

	/* Define 2x2x2 Gauss Quadrature */
	a = 1/sqrt(3);
	rgau[1] = -a;
	rgau[2] = a;
	rgau[3] = a;
	rgau[4] = -a;
	rgau[5] = -a;
	rgau[6] = a;
	rgau[7] = a;
	rgau[8] = -a;
	sgau[1] = -a;
	sgau[2] = -a;
	sgau[3] = a;
	sgau[4] = a;
	sgau[5] = -a;
	sgau[6] = -a;
	sgau[7] = a;
	sgau[8] = a;
	tgau[1] = -a;
	tgau[2] = -a;
	tgau[3] = -a;
	tgau[4] = -a;
	tgau[5] = a;
	tgau[6] = a;
	tgau[7] = a;
	tgau[8] = a;

	/* Inicialization of the element stifness matrix */
	for(i=1;i<=24;i++)
		for(j=1;j<=24;j++)
			ske[i][j] = 0;

	/* Gradient at the 1 point gaussian quadrature */
	J0 = 0;
	J0 = gradH8(xe,ye,ze,0,0,0,N0,J0,dNdx0,dNdy0,dNdz0);

	/* Numerical Integration (2x2x2 Gauss ) normal strain */
	for(i=1;i<=8;i++)
	{
		r=rgau[i];
		s=sgau[i];
		t=tgau[i];

		/* shape functions & their first derivativea */
		J = gradH8(xe,ye,ze,r,s,t,N,J,dNdx,dNdy,dNdz);

		for(j=1;j<=6;j++)
			for(k=1;k<=24;k++)
				B[j][k] = 0;

		for(j=1;j<=8;j++)
		{
			B[1][3*j-2] = dNdx[j];
			B[1][3*j-1] = 0;
			B[1][3*j]   = 0;
			B[2][3*j-2] = 0;
			B[2][3*j-1] = dNdy[j];
			B[2][3*j]   = 0;
			B[3][3*j-2] = 0;
			B[3][3*j-1] = 0;
			B[3][3*j]   = dNdz[j];
		}

		transposta(B,B2);

		multiplica(D,B,B2,ske,J);
	}


	/* numerical integration 2 point gauss shear strain gyz */
	rgau2[1] = -a;
	rgau2[2] = a;
	for(i=1;i<=2;i++)
	{
		r=rgau2[i];
		J = gradH8(xe,ye,ze,r,0,0,N,J,dNdx,dNdy,dNdz);
		for(j=1;j<=6;j++)
			for(k=1;k<=24;k++)
				B[j][k] = 0;
		for(j=1;j<=8;j++)
		{
			B[4][3*j-2]=0;
			B[4][3*j-1]=dNdz[j];
			B[4][3*j]=dNdy[j];
		}

		transposta(B,B2);
		multiplica2(D,B,B2,ske,J);
	}

	/* numerical integration 2 point gauss shear strain gzx */
	sgau2[1] = -a;
	sgau2[2] = a;
	for(i=1;i<=2;i++)
	{
		s=sgau2[i];
		J = gradH8(xe,ye,ze,0,s,0,N,J,dNdx,dNdy,dNdz);
		for(j=1;j<=6;j++)
			for(k=1;k<=24;k++)
				B[j][k] = 0;
		for(j=1;j<=8;j++)
		{
			B[5][3*j-2]=dNdz[j];
			B[5][3*j-1]=0;
			B[5][3*j]=dNdx[j];
		}

		transposta(B,B2);
		multiplica2(D,B,B2,ske,J);
	}

	/* numerical integration 2 point gauss shear strain gxy */
	tgau2[1] = -a;
	tgau2[2] = a;
	for(i=1;i<=2;i++)
	{
		t=tgau2[i];
		J = gradH8(xe,ye,ze,0,0,t,N,J,dNdx,dNdy,dNdz);

		for(j=1;j<=6;j++)
			for(k=1;k<=24;k++)
				B[j][k] = 0;

		for(j=1;j<=8;j++)
		{
			B[6][3*j-2]=dNdy[j];
			B[6][3*j-1]=dNdx[j];
			B[6][3*j]=0;
		}

		transposta(B,B2);
		multiplica2(D,B,B2,ske,J);
	}

	free_vector(ze,1,8);
	free_vector(ye,1,8);
	free_vector(xe,1,8);
	free_vector(dNdx,1,8);
	free_vector(dNdy,1,8);
	free_vector(dNdz,1,8);
	free_vector(N,1,8);
	free_vector(dNdx0,1,8);
	free_vector(dNdy0,1,8);
	free_vector(dNdz0,1,8);
	free_vector(N0,1,8);
	free_matrix(D,1,6,1,6);
	free_vector(rgau,1,8);
	free_vector(sgau,1,8);
	free_vector(tgau,1,8);
	free_vector(rgau2,1,2);
	free_vector(sgau2,1,2);
	free_vector(tgau2,1,2);
	free_matrix(B,1,6,1,24);
	free_matrix(B2,1,24,1,6);

} // Fim SKE

void transposta(float **B,float **B2)
{
	int i,j;

	for(i=1;i<=6;i++)
		for(j=1;j<=24;j++)
			B2[j][i] = B[i][j];
} // Fim transposta

float gradH8(float *xe,float *ye,float *ze,float r,float s,float t,float *N,float J,float *dNdx,float *dNdy,float *dNdz)
{
	float *dNdr, *dNds, *dNdt, **jacmat, **jacinv;
	int i,j;

	dNds = vector(1,8);
	dNdr = vector(1,8);
	dNdt = vector(1,8);
	jacmat = matrix(1,3,1,3);
	jacinv = matrix(1,3,1,3);

	N[1] = (1-r)*(1-s)*(1-t)/8;
	N[2] = (1+r)*(1-s)*(1-t)/8;
	N[3] = (1+r)*(1+s)*(1-t)/8;
	N[4] = (1-r)*(1+s)*(1-t)/8;
	N[5] = (1-r)*(1-s)*(1+t)/8;
	N[6] = (1+r)*(1-s)*(1+t)/8;
	N[7] = (1+r)*(1+s)*(1+t)/8;
	N[8] = (1-r)*(1+s)*(1+t)/8;

	dNdr[1] = -(1-s)*(1-t)/8;
	dNdr[2] =  (1-s)*(1-t)/8;
	dNdr[3] =  (1+s)*(1-t)/8;
	dNdr[4] = -(1+s)*(1-t)/8;
	dNdr[5] = -(1-s)*(1+t)/8;
	dNdr[6] =  (1-s)*(1+t)/8;
	dNdr[7] =  (1+s)*(1+t)/8;
	dNdr[8] = -(1+s)*(1+t)/8;

	dNds[1] = -(1-r)*(1-t)/8;
	dNds[2] = -(1+r)*(1-t)/8;
	dNds[3] =  (1+r)*(1-t)/8;
	dNds[4] =  (1-r)*(1-t)/8;
	dNds[5] = -(1-r)*(1+t)/8;
	dNds[6] = -(1+r)*(1+t)/8;
	dNds[7] =  (1+r)*(1+t)/8;
	dNds[8] =  (1-r)*(1+t)/8;

	dNdt[1] = -(1-r)*(1-s)/8;
	dNdt[2] = -(1+r)*(1-s)/8;
	dNdt[3] = -(1+r)*(1+s)/8;
	dNdt[4] = -(1-r)*(1+s)/8;
	dNdt[5] =  (1-r)*(1-s)/8;
	dNdt[6] =  (1+r)*(1-s)/8;
	dNdt[7] =  (1+r)*(1+s)/8;
	dNdt[8] =  (1-r)*(1+s)/8;

	for(i=1;i<=3;i++)
		for(j=1;j<=3;j++)
			jacmat[i][j] = 0;

	for(i=1;i<=3;i++)
		for(j=1;j<=3;j++)
			jacinv[i][j] = 0;


	for(i=1;i<=8;i++)
	{
		jacmat[1][1] = jacmat[1][1] + dNdr[i]*xe[i];
		jacmat[1][2] = jacmat[1][2] + dNdr[i]*ye[i];
		jacmat[1][3] = jacmat[1][3] + dNdr[i]*ze[i];
		jacmat[2][1] = jacmat[2][1] + dNds[i]*xe[i];
		jacmat[2][2] = jacmat[2][2] + dNds[i]*ye[i];
		jacmat[2][3] = jacmat[2][3] + dNds[i]*ze[i];
		jacmat[3][1] = jacmat[3][1] + dNdt[i]*xe[i];
		jacmat[3][2] = jacmat[3][2] + dNdt[i]*ye[i];
		jacmat[3][3] = jacmat[3][3] + dNdt[i]*ze[i];
	}

	inversa(jacinv,jacmat);

	J = determinante(jacmat);

	for(i=1;i<=8;i++)
	{
		dNdx[i] = jacinv[1][1]*dNdr[i]+jacinv[1][2]*dNds[i]+jacinv[1][3]*dNdt[i];
		dNdy[i] = jacinv[2][1]*dNdr[i]+jacinv[2][2]*dNds[i]+jacinv[2][3]*dNdt[i];
		dNdz[i] = jacinv[3][1]*dNdr[i]+jacinv[3][2]*dNds[i]+jacinv[3][3]*dNdt[i];
	}

	free_vector(dNds,1,8);
	free_vector(dNdr,1,8);
	free_vector(dNdt,1,8);
	free_matrix(jacmat,1,3,1,3);
	free_matrix(jacinv,1,3,1,3);

	return(J);
} // Fim gradH8

void inversa(float **A, float **B)
{
	float a,b,c,d,e,f,g,h,i;

	a = B[1][1];
	b = B[1][2];
	c = B[1][3];
	d = B[2][1];
	e = B[2][2];
	f = B[2][3];
	g = B[3][1];
	h = B[3][2];
	i = B[3][3];


	A[1][1] = (-i*e+f*h)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[1][2] = (i*b-c*h)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[1][3] = -(b*f-c*e)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[2][1] = -(-i*d+f*g)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[2][2] = -(i*a-c*g)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[2][3] = (a*f-c*d)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[3][1] = (-d*h+e*g)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[3][2] = (a*h-b*g)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
	A[3][3] = -(a*e-b*d)/(-i*a*e+a*f*h+i*d*b-d*c*h-g*b*f+g*c*e);
}  // Fim inversa

float determinante(float **B)
{
	float a,b,c,d,e,f,g,h,i,det;

	a = B[1][1];
	b = B[1][2];
	c = B[1][3];
	d = B[2][1];
	e = B[2][2];
	f = B[2][3];
	g = B[3][1];
	h = B[3][2];
	i = B[3][3];

	det = i*a*e-a*f*h-i*d*b+d*c*h+g*b*f-g*c*e;

	return(det);
} // Fim determinante

void multiplica(float **A, float **B, float **B2, float **ske, float J)
{
	int i,j;
	float **temp, **temp2;

	temp=matrix(1,24,1,6);
	temp2=matrix(1,24,1,24);


	/* inicialização da matriz temp */

	for(i=1;i<=24;i++)
		for(j=1;j<=6;j++)
			temp[i][j]=0;

	/* inicialização da matriz temp2 */

	for(i=1;i<=24;i++)
		for(j=1;j<=24;j++)
			temp2[i][j]=0;

	MM(B2,A,24,6,6,6,temp);
	MM(temp,B,24,6,6,24,temp2);

	for(i=1;i<=24;i++)
		for(j=1;j<=24;j++)
			ske[i][j] = ske[i][j] + temp2[i][j]*J;

	free_matrix(temp,1,24,1,6);
	free_matrix(temp2,1,24,1,24);

} // Fim multiplica

void multiplica2(float **A, float **B, float **B2, float **ske, float J)
{
	int i,j;
	float **temp, **temp2;

	temp=matrix(1,24,1,6);
	temp2=matrix(1,24,1,24);

	/* inicialização da matriz temp */
	for(i=1;i<=24;i++)
		for(j=1;j<=6;j++)
			temp[i][j]=0;

	/* inicialização da matriz temp2 */
	for(i=1;i<=24;i++)
		for(j=1;j<=24;j++)
			temp2[i][j]=0;


	MM(B2,A,24,6,6,6,temp);
	MM(temp,B,24,6,6,24,temp2);

	for(i=1;i<=24;i++)
		for(j=1;j<=24;j++)
			ske[i][j] = ske[i][j] + temp2[i][j]*J*4;

	free_matrix(temp,1,24,1,6);
	free_matrix(temp2,1,24,1,24);

} // fim multiplica2

void MM(float **A, float **B, int iA, int jA, int iB, int jB, float **C)
{
	int i,j,k;

	for(i=1;i<=iA;i++)
		for(j=1;j<=jB;j++)
			C[i][j] = 0;

	for(i=1;i<=iA;i++)
		for(j=1;j<=jB;j++)
			for(k=1;k<=jA;k++)
				C[i][j] = C[i][j] + A[i][k]*B[k][j];
} // Fim MM


//função que resolve o sistema de equaçoes
void linbcg(unsigned long n, double b[], double x[], double tol, int itmax, int *iter, double *err)
/*
Solves A.x = b for x[1..n], givenb[1..n], by the iterative biconjugate gradient method.
On input x[1..n] should be set to an initial guess of the solution (or all zeros); itol is 1,
specifying which convergence test is applied (see text); itmax is the maximum number
of allowed iterations; and tol is the desired convergence tolerance. On output, x[1..n] is
reset to the improved solution, iter is the number of iterations actually taken, and err is the
estimated error. The matrix A is referenced only through the user-supplied routines atimes,
which computes the product of either A or its transpose on a vector; and asolve, which solves A   x = b or AT  x = b for some preconditioner matrix A (possibly the trivial diagonal part of A).
*/
{
//	void asolve(unsigned long n, double b[], double x[], int itrnsp);
//	double snrm(unsigned long n, double sx[]);

	unsigned long j;
	double ak,akden,bk,bkden,bknum,bnrm;
	double *p,*pp,*r,*rr,*z,*zz; 	//Double precision is a good idea in this routine.

	p=dvector(1,n);
	pp=dvector(1,n);
	r=dvector(1,n);
	rr=dvector(1,n);
	z=dvector(1,n);
	zz=dvector(1,n);

	//Calculate initial residual.
	*iter=0;
	dsprsax(sa,ija,x,r,n);                                  //Input is x[1..n], output is r[1..n];

        for (j=1;j<=n;j++) {
		r[j]=b[j]-r[j];
		rr[j]=r[j];
	}

        bnrm=snrm(n,b);
	asolve(n,r,z,0);				//Input to asolve is r[1..n], output is z[1..n];
							//the final 0 indicates that the matrix A (not
							//its transpose) is to be used.

        while (*iter <= itmax)			//Main loop.
        {
		++(*iter);
		asolve(n,rr,zz,1); 				//Final 1 indicates use of transpose matrix AT.
		for (bknum=0.0,j=1;j<=n;j++) bknum += z[j]*rr[j];

	//Calculate coeficient bk and direction vectors p and pp.
		if (*iter == 1)
                {
			for (j=1;j<=n;j++)
                        {
				p[j]=z[j];
				pp[j]=zz[j];
			}  // fim for
		} // fim if
		else
                {
			bk=bknum/bkden;
			for (j=1;j<=n;j++)
                        {
				p[j]=bk*p[j]+z[j];
				pp[j]=bk*pp[j]+zz[j];
			} // fim for
		} // fim else
		bkden=bknum; 				//Calculate coeficient ak, new iterate x, and new
							//residuals r and rr.
                dsprsax(sa,ija,p,z,n);
		for (akden=0.0,j=1;j<=n;j++) akden += z[j]*pp[j];
		ak=bknum/akden;
                dsprstx(sa,ija,pp,zz,n);
		for (j=1;j<=n;j++)
                {
			x[j] += ak*p[j];
			r[j] -= ak*z[j];
			rr[j] -= ak*zz[j];
		}
		asolve(n,r,z,0);			//Solve A.z = r and check stopping criterion.
		*err=snrm(n,r)/bnrm;
        	if (*err <= tol) break;
	} // fim main loop

	free_dvector(p,1,n);
	free_dvector(pp,1,n);
	free_dvector(r,1,n);
	free_dvector(rr,1,n);
	free_dvector(z,1,n);
	free_dvector(zz,1,n);
} // Fim linbcg


//funçao que multiplica a matriz esparsa indexada pelo vetor x[], resultado é o vetor b[]
void dsprsax(double sa[], unsigned long ija[], double x[], double b[], unsigned long n)
{
  unsigned long i,k;

  if (ija[1] != n+2) nrerror("dsprsax: mismatched vector and matrix");
  for (i=1;i<=n;i++)
  {
	  b[i]=sa[i]*x[i];				//Start with diagonal term.
	  for (k=ija[i];k<=ija[i+1]-1;k++) 		//Loop over off-diagonal terms.
		  b[i] += sa[k]*x[ija[k]];
  }
} // Fim dsprsax


//funçao que multiplica a matriz esparsa indexada transposta pelo vetor x[], resultado é o vetor b[]
void dsprstx(double sa[], unsigned long ija[], double x[], double b[], unsigned long n)
{
  unsigned long i,j,k;

  if (ija[1] != n+2) nrerror("mismatched vector and matrix in sprstx");
  for (i=1;i<=n;i++) b[i]=sa[i]*x[i]; 					//Start with diagonal terms.
  for (i=1;i<=n;i++)							//Loop over off-diagonal terms.
  {
	  for (k=ija[i];k<=ija[i+1]-1;k++)
          {
		  j=ija[k];
		  b[j] += sa[k]*x[i];
	  }
  }
} /* fim dsprstx */


void asolve(unsigned long n, double b[], double x[], int itrnsp)
{
	unsigned long i;
	for(i=1;i<=n;i++) x[i]=(sa[i] != 0.0 ? b[i]/sa[i] : b[i]);
//The matrix Ã is the diagonal part of A, stored in the first n elements of sa.
//Since the transpose matrix has the same diagonal, the ag itrnsp is not used.
} /*fim ASOLVE */


double snrm(unsigned long n, double sx[])
//Compute one of two norms for a vector sx[1..n], as signaled by itol. Used by linbcg.
{
	unsigned long i;
	double ans;

	ans = 0.0;
	for (i=1;i<=n;i++) ans += sx[i]*sx[i];			//Vector magnitude norm.
	return sqrt(ans);
} /* fim snrm */

/////////////////////////////////////////////////////////////////////////////////////
//////////////////////////Funções de alocação do nrutil//////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////

void nrerror(char error_text[])
/* Numerical Recipes standard error handler */
{
	fprintf(stderr,"Numerical Recipes run-time error...\n");
	fprintf(stderr,"%s\n",error_text);
	fprintf(stderr,"...now exiting to system...\n");
	exit(1);
} /* fim nrerror */

float *vector(long nl, long nh)
/* allocate a float vector with subscript range v[nl..nh] */
{
	float *v;

	v=(float *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(float)));
	if (!v) nrerror("allocation failure in vector()");
	return v-nl+NR_END;
} /* fim *vector */

int *ivector(long nl, long nh)
/* allocate an int vector with subscript range v[nl..nh] */
{
	int *v;

	v=(int *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(int)));
	if (!v) nrerror("allocation failure in ivector()");
	return v-nl+NR_END;
} /* fim *ivector */

unsigned char *cvector(long nl, long nh)
/* allocate an unsigned char vector with subscript range v[nl..nh] */
{
	unsigned char *v;

	v=(unsigned char *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(unsigned char)));
	if (!v) nrerror("allocation failure in cvector()");
	return v-nl+NR_END;
} /* fim *cvector */

unsigned long *lvector(long nl, long nh)
/* allocate an unsigned long vector with subscript range v[nl..nh] */
{
	unsigned long *v;

	v=(unsigned long *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(long)));
	if (!v) nrerror("allocation failure in lvector()");
	return v-nl+NR_END;
} /* fim *lvector */

double *dvector(long nl, long nh)
/* allocate a double vector with subscript range v[nl..nh] */
{
	double *v;

	v=(double *)malloc((size_t) ((nh-nl+1+NR_END)*sizeof(double)));
	if (!v) nrerror("allocation failure in dvector()");
	return v-nl+NR_END;
} /* fim dvector */

float **matrix(long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t)((nrow+NR_END)*sizeof(float*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(float *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
} /* fim **matrix */

double **dmatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a double matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	double **m;

	/* allocate pointers to rows */
	m=(double **) malloc((size_t)((nrow+NR_END)*sizeof(double*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;

	/* allocate rows and set pointers to them */
	m[nrl]=(double *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(double)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
} /* fim **dmatrix */

int **imatrix(long nrl, long nrh, long ncl, long nch)
/* allocate a int matrix with subscript range m[nrl..nrh][ncl..nch] */
{
	long i, nrow=nrh-nrl+1,ncol=nch-ncl+1;
	int **m;

	/* allocate pointers to rows */
	m=(int **) malloc((size_t)((nrow+NR_END)*sizeof(int*)));
	if (!m) nrerror("allocation failure 1 in matrix()");
	m += NR_END;
	m -= nrl;


	/* allocate rows and set pointers to them */
	m[nrl]=(int *) malloc((size_t)((nrow*ncol+NR_END)*sizeof(int)));
	if (!m[nrl]) nrerror("allocation failure 2 in matrix()");
	m[nrl] += NR_END;
	m[nrl] -= ncl;

	for(i=nrl+1;i<=nrh;i++) m[i]=m[i-1]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
} /* fim *imatrix */

float **submatrix(float **a, long oldrl, long oldrh, long oldcl, long oldch, long newrl, long newcl)
/* point a submatrix [newrl..][newcl..] to a[oldrl..oldrh][oldcl..oldch] */
{
	long i,j,nrow=oldrh-oldrl+1,ncol=oldcl-newcl;
	float **m;

	/* allocate array of pointers to rows */
	m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
	if (!m) nrerror("allocation failure in submatrix()");
	m += NR_END;
	m -= newrl;

	/* set pointers to rows */
	for(i=oldrl,j=newrl;i<=oldrh;i++,j++) m[j]=a[i]+ncol;

	/* return pointer to array of pointers to rows */
	return m;
} /* fim **submatrix */

float **convert_matrix(float *a, long nrl, long nrh, long ncl, long nch)
/* allocate a float matrix m[nrl..nrh][ncl..nch] that points to the matrix
declared in the standard C manner as a[nrow][ncol], where nrow=nrh-nrl+1
and ncol=nch-ncl+1. The routine should be called with the address
&a[0][0] as the first argument. */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1;
	float **m;

	/* allocate pointers to rows */
	m=(float **) malloc((size_t) ((nrow+NR_END)*sizeof(float*)));
	if (!m) nrerror("allocation failure in convert_matrix()");
	m += NR_END;
	m -= nrl;

	/* set pointers to rows */
	m[nrl]=a-ncl;
	for(i=1,j=nrl+1;i<nrow;i++,j++) m[j]=m[j-1]+ncol;
	/* return pointer to array of pointers to rows */
	return m;
} /* fim **convert_matrix */

float ***f3tensor(long nrl, long nrh, long ncl, long nch, long ndl, long ndh)
/* allocate a float 3tensor with range t[nrl..nrh][ncl..nch][ndl..ndh] */
{
	long i,j,nrow=nrh-nrl+1,ncol=nch-ncl+1,ndep=ndh-ndl+1;
	float ***t;

	/* allocate pointers to pointers to rows */
	t=(float ***) malloc((size_t)((nrow+NR_END)*sizeof(float**)));
	if (!t) nrerror("allocation failure 1 in f3tensor()");
	t += NR_END;
	t -= nrl;

	/* allocate pointers to rows and set pointers to them */
	t[nrl]=(float **) malloc((size_t)((nrow*ncol+NR_END)*sizeof(float*)));
	if (!t[nrl]) nrerror("allocation failure 2 in f3tensor()");
	t[nrl] += NR_END;
	t[nrl] -= ncl;

	/* allocate rows and set pointers to them */
	t[nrl][ncl]=(float *) malloc((size_t)((nrow*ncol*ndep+NR_END)*sizeof(float)));
	if (!t[nrl][ncl]) nrerror("allocation failure 3 in f3tensor()");
	t[nrl][ncl] += NR_END;
	t[nrl][ncl] -= ndl;

	for(j=ncl+1;j<=nch;j++) t[nrl][j]=t[nrl][j-1]+ndep;
	for(i=nrl+1;i<=nrh;i++) {
		t[i]=t[i-1]+ncol;
		t[i][ncl]=t[i-1][ncl]+ncol*ndep;
		for(j=ncl+1;j<=nch;j++) t[i][j]=t[i][j-1]+ndep;
	}

	/* return pointer to array of pointers to rows */
	return t;
} /* fim ***f3tensor */

void free_vector(float *v, long nl, long nh)
/* free a float vector allocated with vector() */
{
	free((char*) (v+nl-NR_END));
}

void free_ivector(int *v, long nl, long nh)
/* free an int vector allocated with ivector() */
{
	free((char*) (v+nl-NR_END));
}

void free_cvector(unsigned char *v, long nl, long nh)
/* free an unsigned char vector allocated with cvector() */
{
	free((char*) (v+nl-NR_END));
}

void free_lvector(unsigned long *v, long nl, long nh)
/* free an unsigned long vector allocated with lvector() */
{
	free((char*) (v+nl-NR_END));
}

void free_dvector(double *v, long nl, long nh)
/* free a double vector allocated with dvector() */
{
	free((char*) (v+nl-NR_END));
}

void free_matrix(float **m, long nrl, long nrh, long ncl, long nch)
/* free a float matrix allocated by matrix() */
{
	free((char*) (m[nrl]+ncl-NR_END));
	free((char*) (m+nrl-NR_END));
}

void free_dmatrix(double **m, long nrl, long nrh, long ncl, long nch)
/* free a double matrix allocated by dmatrix() */
{
	free((char*) (m[nrl]+ncl-NR_END));
	free((char*) (m+nrl-NR_END));
}

void free_imatrix(int **m, long nrl, long nrh, long ncl, long nch)
/* free an int matrix allocated by imatrix() */
{
	free((char*) (m[nrl]+ncl-NR_END));
	free((char*) (m+nrl-NR_END));
}

void free_submatrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a submatrix allocated by submatrix() */
{
	free((char*) (b+nrl-NR_END));
}

void free_convert_matrix(float **b, long nrl, long nrh, long ncl, long nch)
/* free a matrix allocated by convert_matrix() */
{
	free((char*) (b+nrl-NR_END));
}

void free_f3tensor(float ***t, long nrl, long nrh, long ncl, long nch,
	long ndl, long ndh)
/* free a float f3tensor allocated by f3tensor() */
{
	free((char*) (t[nrl][ncl]+ndl-NR_END));
	free((char*) (t[nrl]+ncl-NR_END));
	free((char*) (t+nrl-NR_END));
} /* fim das funções de liberação de memória */

/////////////////  FIM  END  FIN  ////////////////////////////////////////
