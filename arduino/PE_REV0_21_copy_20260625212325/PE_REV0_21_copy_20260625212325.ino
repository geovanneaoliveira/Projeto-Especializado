// ============================================================
//  Controle de Motores com FreeRTOS (Otimizado)
//  Hardware: RAMPS 1.4/1.6 + RepRap Smart Controller (Mega 2560)
// ============================================================

#include <Arduino_FreeRTOS.h>
#include <queue.h>
#include <LiquidCrystal.h>

// ============================================================
//  Pinos — Display e Botões
// ============================================================
LiquidCrystal lcd(16, 17, 23, 25, 27, 29);

const int pinoEncoderA     = 31;
const int pinoEncoderB     = 33;
const int pinoBotaoEncoder = 35;  
const int pinoBotaoExtra   = 41;  

// ============================================================
//  Pinos — Motores Z (Esteira) e X
// ============================================================
const int stepPinZ = 46;
const int dirPinZ  = 48;
const int enPinZ   = 62;

const int stepPinX = 54;
const int dirPinX  = 55;
const int enPinX   = 38;

const int PASSOS_MEIA_VOLTA = 1600;

// ============================================================
//  Tipos de Comando e Fila
// ============================================================
typedef enum {
  CMD_LIGA_Z,       
  CMD_DESLIGA_Z,    
  CMD_TOGGLE_Z,     
  CMD_DISPARA_X,    
  CMD_VELOCIDADE    
} TipoComando;

typedef struct {
  TipoComando tipo;
  int         valor;  
} Comando;

QueueHandle_t filaComandos;

volatile bool motorZLigado = false;
volatile int  motorZAtraso = 650; 

// ============================================================
//  Protótipos
// ============================================================
void TaskSerial   (void *pvParameters);
void TaskControle (void *pvParameters);
void TaskMotorZ   (void *pvParameters);

void executarMovimentoX(int velocidadePct);
void atualizarLCD(bool ligado, int velocidadePct);

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  pinMode(stepPinZ, OUTPUT); pinMode(dirPinZ, OUTPUT); pinMode(enPinZ, OUTPUT);
  digitalWrite(enPinZ, HIGH); 
  digitalWrite(dirPinZ, LOW);

  pinMode(stepPinX, OUTPUT); pinMode(dirPinX, OUTPUT); pinMode(enPinX, OUTPUT);
  digitalWrite(enPinX, HIGH); 

  pinMode(pinoEncoderA,     INPUT_PULLUP);
  pinMode(pinoEncoderB,     INPUT_PULLUP);
  pinMode(pinoBotaoEncoder, INPUT_PULLUP);
  pinMode(pinoBotaoExtra,   INPUT_PULLUP);

  lcd.begin(20, 4);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Controle de Motores ");
  lcd.setCursor(0, 1); lcd.print("Status: PARADO      ");
  lcd.setCursor(0, 2); lcd.print("Velocidade: 50%     ");
  lcd.setCursor(0, 3); lcd.print("Aguardando...       ");

  filaComandos = xQueueCreate(10, sizeof(Comando));

  if (filaComandos == NULL) {
    Serial.println(F("ERRO: fila falhou!"));
    lcd.setCursor(0, 3); lcd.print("ERRO: fila falhou!  ");
    while (1); 
  }

  // CORREÇÃO 1: Pilhas aumentadas para 256 words (512 bytes) para evitar Stack Overflow
  xTaskCreate(TaskSerial,   "Serial",   256, NULL, 2, NULL); 
  xTaskCreate(TaskControle, "Controle", 256, NULL, 2, NULL); 
  xTaskCreate(TaskMotorZ,   "MotorZ",   128, NULL, 2, NULL); 

  Serial.println(F("=== Controle de Motores Iniciado ==="));
}

void loop() {}

// ============================================================
//  TASK 1: TaskSerial
// ============================================================
void TaskSerial(void *pvParameters) {
  (void)pvParameters;

  // CORREÇÃO 2: Substituição da classe String por um array de char (Buffer)
  char bufferVelocidade[5];
  uint8_t indexBuf = 0;

  int ultimoA       = digitalRead(pinoEncoderA);
  int ultimoBtn     = HIGH;
  int estadoBtn     = HIGH;
  unsigned long tBtn = 0;

  int ultimoBtnX    = HIGH;
  int estadoBtnX    = HIGH; // Novo rastreador de estado para evitar laço travante
  unsigned long tBtnX = 0;
  
  int velocidadeEncoder = 50;
  Comando cmd;

  for (;;) {
    
    // --- Leitura Serial Segura ---
    while (Serial.available() > 0) {
      char c = Serial.read();

      if (c == 'a' || c == 'A') {
        cmd = {CMD_LIGA_Z, 0};
        xQueueSend(filaComandos, &cmd, portMAX_DELAY);
        Serial.println(F("[SERIAL] INICIAR"));

      } else if (c == 's' || c == 'S') {
        cmd = {CMD_DESLIGA_Z, 0};
        xQueueSend(filaComandos, &cmd, portMAX_DELAY);
        Serial.println(F("[SERIAL] PARAR"));

      } else if (c == 'd' || c == 'D') {
        cmd = {CMD_DISPARA_X, 0};
        xQueueSend(filaComandos, &cmd, portMAX_DELAY);
        Serial.println(F("[SERIAL] EIXO X"));

      } else if (isDigit(c)) {
        if (indexBuf < 4) { // Limita a 4 caracteres (ex: "100\0")
          bufferVelocidade[indexBuf++] = c;
          bufferVelocidade[indexBuf] = '\0';
        }

      } else if (c == '\n' || c == '\r') {
        if (indexBuf > 0) {
          int vel = atoi(bufferVelocidade); // Converte array de char para int
          if (vel >= 0 && vel <= 100) {
            cmd = {CMD_VELOCIDADE, vel};
            xQueueSend(filaComandos, &cmd, portMAX_DELAY);
            Serial.print(F("[SERIAL] VELOCIDADE "));
            Serial.println(vel);
          } else {
            Serial.println(F("[ERRO] 0 a 100"));
          }
          indexBuf = 0; // Reseta o buffer
        }
      }
    }

    // --- Encoder ---
    int curA = digitalRead(pinoEncoderA);
    if (curA != ultimoA && curA == LOW) {
      if (digitalRead(pinoEncoderB) != curA) velocidadeEncoder -= 5;
      else                                   velocidadeEncoder += 5;
      velocidadeEncoder = constrain(velocidadeEncoder, 0, 100);
      cmd = {CMD_VELOCIDADE, velocidadeEncoder};
      xQueueSend(filaComandos, &cmd, 0); 
    }
    ultimoA = curA;

    // --- Botão Encoder ---
    int lBtn = digitalRead(pinoBotaoEncoder);
    if (lBtn != ultimoBtn) tBtn = millis();
    if ((millis() - tBtn) > 50 && lBtn != estadoBtn) {
      estadoBtn = lBtn;
      if (estadoBtn == LOW) {
        cmd = {CMD_TOGGLE_Z, 0};
        xQueueSend(filaComandos, &cmd, 0);
      }
    }
    ultimoBtn = lBtn;

    // --- Botão Extra (Eixo X) - CORREÇÃO 3: Sem laço 'while' ---
    int lBtnX = digitalRead(pinoBotaoExtra);
    if (lBtnX != ultimoBtnX) tBtnX = millis();
    if ((millis() - tBtnX) > 50 && lBtnX != estadoBtnX) {
      estadoBtnX = lBtnX;
      if (estadoBtnX == LOW) { // Dispara apenas na borda de descida (quando pressionado)
        cmd = {CMD_DISPARA_X, 0};
        xQueueSend(filaComandos, &cmd, 0);
      }
    }
    ultimoBtnX = lBtnX;

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// ============================================================
//  TASK 2: TaskControle
// ============================================================
void TaskControle(void *pvParameters) {
  (void)pvParameters;

  Comando cmdRecebido;
  bool ligado     = false;
  int  velocidade = 50;

  motorZLigado = false;
  motorZAtraso = 650;
  digitalWrite(enPinZ, HIGH);
  digitalWrite(enPinX, HIGH);

  atualizarLCD(ligado, velocidade);

  for (;;) {
    xQueueReceive(filaComandos, &cmdRecebido, portMAX_DELAY);

    switch (cmdRecebido.tipo) {
      case CMD_LIGA_Z:
        ligado       = true;
        motorZLigado = true;
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_DESLIGA_Z:
        ligado       = false;
        motorZLigado = false;
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_TOGGLE_Z:
        ligado       = !ligado;
        motorZLigado = ligado;
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_DISPARA_X:
        executarMovimentoX(velocidade);
        Serial.println(F("[CONTROLE] Eixo X concluido"));
        break;

      case CMD_VELOCIDADE:
        velocidade   = cmdRecebido.valor;
        motorZAtraso = map(velocidade, 0, 100, 1200, 100);
        atualizarLCD(ligado, velocidade);
        break;
    }
  }
}

// ============================================================
//  TASK 3: TaskMotorZ
// ============================================================
void TaskMotorZ(void *pvParameters) {
  (void)pvParameters;

  for (;;) {
    if (motorZLigado) {
      digitalWrite(enPinZ, LOW);
      digitalWrite(stepPinZ, HIGH);
      delayMicroseconds(motorZAtraso);
      digitalWrite(stepPinZ, LOW);
      delayMicroseconds(motorZAtraso);
      taskYIELD();
    } else {
      digitalWrite(enPinZ, HIGH);
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

// ============================================================
//  Sequência de movimento do Eixo X
// ============================================================
void executarMovimentoX(int velocidadePct) {
  const int VEL_X = 150; 

  digitalWrite(enPinX, LOW);

  digitalWrite(dirPinX, HIGH);
  for (int i = 0; i < PASSOS_MEIA_VOLTA; i++) {
    digitalWrite(stepPinX, HIGH);
    delayMicroseconds(2);       
    digitalWrite(stepPinX, LOW);
    delayMicroseconds(VEL_X);
    taskYIELD(); 
  }

  vTaskDelay(pdMS_TO_TICKS(100));

  digitalWrite(dirPinX, LOW);
  for (int i = 0; i < PASSOS_MEIA_VOLTA; i++) {
    digitalWrite(stepPinX, HIGH);
    delayMicroseconds(2);
    digitalWrite(stepPinX, LOW);
    delayMicroseconds(VEL_X);
    taskYIELD(); 
  }

  digitalWrite(enPinX, HIGH); 
}

// ============================================================
//  Atualiza o LCD
// ============================================================
void atualizarLCD(bool ligado, int velocidadePct) {
  lcd.setCursor(0, 0); lcd.print("Controle de Motores ");

  lcd.setCursor(0, 1);
  lcd.print("Status: ");
  lcd.print(ligado ? "LIGADO   " : "PARADO   ");

  lcd.setCursor(0, 2);
  lcd.print("Velocidade: ");
  lcd.print(velocidadePct);
  lcd.print("%   ");

  int atraso = map(velocidadePct, 0, 100, 1200, 100);
  lcd.setCursor(0, 3);
  lcd.print("Pulso: ");
  lcd.print(atraso);
  lcd.print(" us    ");
}