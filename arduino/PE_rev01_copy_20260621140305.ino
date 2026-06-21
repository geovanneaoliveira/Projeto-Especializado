#include <LiquidCrystal.h>

// --- Pinos do Display (RepRap Smart Controller na RAMPS 1.4/1.6) ---
// O adaptador joga os pinos lógicos para estas portas específicas do Mega
LiquidCrystal lcd(16, 17, 23, 25, 27, 29);

// --- Pinos do Encoder Rotativo ---
const int pinoEncoderA = 31;
const int pinoEncoderB = 33;
const int pinoBotaoEncoder = 35;

// --- Pinos dos Motores (Eixo Z) ---
const int stepPin = 46; 
const int dirPin = 48; 
const int enPin = 62;

// --- Variáveis de Estado ---
int atraso = 1000;         // Limites solicitados: 100 a 1200
bool motorLigado = false;  // Começa parado

// Variáveis de controle do Encoder e filtro Debounce
int ultimoEstadoA = HIGH;
int estadoBotao;
int ultimoEstadoBotao = HIGH;
unsigned long ultimoTempoDebounce = 0;
unsigned long tempoDebounce = 50; // milissegundos

// Controle de atualização da tela
bool telaPrecisaAtualizar = true;
unsigned long ultimaAtualizacaoTela = 0;

void setup() {
  // Configuração dos motores
  pinMode(stepPin, OUTPUT); 
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  
  // Inicia com o driver desabilitado (HIGH corta a corrente nos DRV8825)
  digitalWrite(enPin, HIGH); 
  digitalWrite(dirPin, LOW); 
  
  // Configuração do painel
  pinMode(pinoEncoderA, INPUT_PULLUP);
  pinMode(pinoEncoderB, INPUT_PULLUP);
  pinMode(pinoBotaoEncoder, INPUT_PULLUP);
  
  // Inicialização do LCD
  lcd.begin(20, 4);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Controle de Motores");
  
  ultimoEstadoA = digitalRead(pinoEncoderA);
  atualizarTela();
}

void loop() {
  lerEncoder();
  lerBotao();
  
  // --- Geração do Sinal PWM (Apenas se o sistema estiver ligado) ---
  if (motorLigado) {
    digitalWrite(enPin, LOW); // Habilita o driver
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(atraso); 
    digitalWrite(stepPin, LOW);
    delayMicroseconds(atraso);
  } else {
    digitalWrite(enPin, HIGH); // Corta energia para não superaquecer os motores parados
  }

  // --- Atualização Assíncrona da Tela ---
  // Escrever no LCD é um processo lento. Para não prejudicar o sincronismo
  // dos passos, a tela só é desenhada a cada 200ms e apenas se houver mudanças.
  if (telaPrecisaAtualizar && (millis() - ultimaAtualizacaoTela > 200)) {
    atualizarTela();
    telaPrecisaAtualizar = false;
    ultimaAtualizacaoTela = millis();
  }
}

void lerEncoder() {
  int estadoAtualA = digitalRead(pinoEncoderA);
  
  // Se o pino A caiu para nível lógico baixo (detectou um "clique" do botão giratório)
  if (estadoAtualA != ultimoEstadoA && estadoAtualA == LOW) {
    // Avalia o pino B para descobrir a direção do giro
    if (digitalRead(pinoEncoderB) != estadoAtualA) {
      atraso -= 50; // Sentido Horário: diminui o atraso (aumenta a velocidade)
    } else {
      atraso += 50; // Sentido Anti-horário: aumenta o atraso (diminui a velocidade)
    }
    
    // Trava os valores nos limites absolutos estabelecidos
    if (atraso < 100) atraso = 100;
    if (atraso > 1200) atraso = 1200;
    
    telaPrecisaAtualizar = true;
  }
  ultimoEstadoA = estadoAtualA;
}

void lerBotao() {
  int leitura = digitalRead(pinoBotaoEncoder);
  
  // Filtro de ruído (Debounce) mecânico do botão
  if (leitura != ultimoEstadoBotao) {
    ultimoTempoDebounce = millis();
  }
  
  if ((millis() - ultimoTempoDebounce) > tempoDebounce) {
    if (leitura != estadoBotao) {
      estadoBotao = leitura;
      
      // Inverte o estado de LIGADO/DESLIGADO quando o botão for apertado
      if (estadoBotao == LOW) {
        motorLigado = !motorLigado;
        telaPrecisaAtualizar = true;
      }
    }
  }
  ultimoEstadoBotao = leitura;
}

void atualizarTela() {
  // Linha 2
  lcd.setCursor(0, 1);
  lcd.print("Status: ");
  if (motorLigado) {
    lcd.print("LIGADO   ");
  } else {
    lcd.print("PARADO   ");
  }
  
  // Linha 3
  lcd.setCursor(0, 2);
  lcd.print("Pulso: ");
  lcd.print(atraso);
  lcd.print(" us    "); 
  
  // Linha 4: Mapeia o valor invertido para criar um percentual intuitivo (0% a 100%)
  lcd.setCursor(0, 3);
  lcd.print("Veloc: ");
  int porcentagem = map(atraso, 1200, 100, 0, 100);
  lcd.print(porcentagem);
  lcd.print("%   ");
}