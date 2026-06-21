// ============================================================
//  Controle de Motores com FreeRTOS
//  Hardware: RAMPS 1.4/1.6 + RepRap Smart Controller (Mega 2560)
//
//  Baseado no padrão dos labs:
//  - 2 tasks: TaskSerial (produtora) e TaskControle (consumidora)
//  - 1 fila de comandos (Queue)
//  - Scheduler inicia automaticamente ao fim do setup()
//  - loop() vazio
//
//  Comandos Serial (115200 baud):
//    a         → Liga esteira (motor Z)
//    s         → Para esteira
//    d         → Dispara movimento eixo X (meia volta ida + volta)
//    0-100 + ENTER → Define velocidade em porcentagem (ex: 75 + ENTER)
// ============================================================

#include <Arduino_FreeRTOS.h>
#include <queue.h>
#include <LiquidCrystal.h>

// ============================================================
//  Pinos — Display
// ============================================================
LiquidCrystal lcd(16, 17, 23, 25, 27, 29);

// ============================================================
//  Pinos — Encoder e botões físicos
// ============================================================
const int pinoEncoderA     = 31;
const int pinoEncoderB     = 33;
const int pinoBotaoEncoder = 35;  // botão do encoder → toggle esteira
const int pinoBotaoExtra   = 41;  // botão extra      → dispara eixo X

// ============================================================
//  Pinos — Motor Z (Esteira)
// ============================================================
const int stepPinZ = 46;
const int dirPinZ  = 48;
const int enPinZ   = 62;

// ============================================================
//  Pinos — Motor X
// ============================================================
const int stepPinX = 54;
const int dirPinX  = 55;
const int enPinX   = 38;

// Passos por meia volta (NEMA17, microstepping 1/16)
const int PASSOS_MEIA_VOLTA = 3200;

// ============================================================
//  Tipos de comando (igual ao padrão do lab mini-CLP)
// ============================================================
typedef enum {
  CMD_LIGA_Z,       // liga a esteira
  CMD_DESLIGA_Z,    // para a esteira
  CMD_TOGGLE_Z,     // inverte estado da esteira (botão físico)
  CMD_DISPARA_X,    // executa sequência do eixo X
  CMD_VELOCIDADE    // campo valor = 0 a 100 (%)
} TipoComando;

typedef struct {
  TipoComando tipo;
  int         valor;  // usado em CMD_VELOCIDADE
} Comando;

// ============================================================
//  Handle da fila — único ponto de comunicação entre tasks
// ============================================================
QueueHandle_t filaComandos;

// ============================================================
//  Estado do motor Z — compartilhado entre TaskControle e TaskMotorZ
//  volatile garante que o compilador não otimize as leituras
// ============================================================
volatile bool motorZLigado = false;
volatile int  motorZAtraso = 650; // µs entre steps

// ============================================================
//  Protótipos
// ============================================================
void TaskSerial   (void *pvParameters);
void TaskControle (void *pvParameters);
void TaskMotorZ   (void *pvParameters); // gera pulsos STEP independentemente

// Rotina de movimento do eixo X (chamada dentro da TaskControle)
void executarMovimentoX(int velocidadePct);

// Atualiza o display LCD (chamada dentro da TaskControle)
void atualizarLCD(bool ligado, int velocidadePct);

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  // --- Motores ---
  pinMode(stepPinZ, OUTPUT); pinMode(dirPinZ, OUTPUT); pinMode(enPinZ, OUTPUT);
  digitalWrite(enPinZ, HIGH); // desabilitado
  digitalWrite(dirPinZ, LOW);

  pinMode(stepPinX, OUTPUT); pinMode(dirPinX, OUTPUT); pinMode(enPinX, OUTPUT);
  digitalWrite(enPinX, HIGH); // desabilitado

  // --- Painel ---
  pinMode(pinoEncoderA,     INPUT_PULLUP);
  pinMode(pinoEncoderB,     INPUT_PULLUP);
  pinMode(pinoBotaoEncoder, INPUT_PULLUP);
  pinMode(pinoBotaoExtra,   INPUT_PULLUP);

  // --- LCD ---
  lcd.begin(20, 4);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("Controle de Motores ");
  lcd.setCursor(0, 1); lcd.print("Status: PARADO      ");
  lcd.setCursor(0, 2); lcd.print("Velocidade: 50%     ");
  lcd.setCursor(0, 3); lcd.print("Aguardando...       ");

  // --- Fila com capacidade para 10 comandos ---
  // Cada item tem 5 bytes (TipoComando=1 + int=4)
  // 10 itens × 5 bytes = 50 bytes de heap + overhead da fila (~80 bytes)
  filaComandos = xQueueCreate(10, sizeof(Comando));

  if (filaComandos == NULL) {
    Serial.println(F("ERRO: nao foi possivel criar a fila!"));
    lcd.setCursor(0, 3); lcd.print("ERRO: fila falhou!  ");
    while (1); // trava — não deve acontecer no Mega com heap suficiente
  }

  // --- Tasks ---
  // Stack em words (1 word = 2 bytes no AVR)
  // TaskSerial: acumula string, sem delayMicroseconds → 128 words OK
  // TaskControle: gera pulsos de step, chama funções maiores → 192 words
  xTaskCreate(TaskSerial,   "Serial",   128, NULL, 2, NULL); // prioridade média
  xTaskCreate(TaskControle, "Controle", 128, NULL, 2, NULL); // prioridade média
  xTaskCreate(TaskMotorZ,   "MotorZ",    80, NULL, 1, NULL); // prioridade BAIXA

  // No Arduino Mega com Arduino_FreeRTOS, o scheduler inicia
  // AUTOMATICAMENTE ao sair do setup(). Não chamar vTaskStartScheduler().

  Serial.println(F("=== Controle de Motores ==="));
  Serial.println(F("  a         -> Liga esteira"));
  Serial.println(F("  s         -> Para esteira"));
  Serial.println(F("  d         -> Dispara eixo X"));
  Serial.println(F("  0-100 + ENTER -> Define velocidade"));
  Serial.println(F("==========================="));
}

// ============================================================
//  LOOP — vazio. O FreeRTOS assume o controle.
// ============================================================
void loop() {}

// ============================================================
//  TASK 1: TaskSerial  (Prioridade 1 — baixa)
//
//  Produtora de comandos.
//  Lê caracteres do Serial e envia Comando para a fila.
//  Também lê o encoder e os botões físicos.
//
//  Padrão do lab: caracteres individuais para a/s/d,
//  número acumulado + ENTER para velocidade.
// ============================================================
void TaskSerial(void *pvParameters) {
  (void)pvParameters;

  String inputString = "";

  // Variáveis do encoder
  int  ultimoA       = digitalRead(pinoEncoderA);
  int  ultimoBtn     = HIGH;
  int  estadoBtn     = HIGH;
  unsigned long tBtn = 0;

  int  ultimoBtnX    = HIGH;
  unsigned long tBtnX = 0;

  // Rastreia velocidade local para o encoder (0-100%)
  int velocidadeEncoder = 50;

  Comando cmd;

  for (;;) {

    // --- Leitura do Serial ---
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
        inputString += c; // acumula dígitos

      } else if (c == '\n' || c == '\r') {
        if (inputString.length() > 0) {
          int vel = inputString.toInt();
          if (vel >= 0 && vel <= 100) {
            cmd = {CMD_VELOCIDADE, vel};
            xQueueSend(filaComandos, &cmd, portMAX_DELAY);
            Serial.print(F("[SERIAL] VELOCIDADE "));
            Serial.println(vel);
          } else {
            Serial.println(F("[ERRO] Velocidade deve ser 0 a 100"));
          }
          inputString = "";
        }
      }
    }

    // --- Encoder rotativo → velocidade ---
    int curA = digitalRead(pinoEncoderA);
    if (curA != ultimoA && curA == LOW) {
      if (digitalRead(pinoEncoderB) != curA) velocidadeEncoder -= 5;
      else                                    velocidadeEncoder += 5;
      velocidadeEncoder = constrain(velocidadeEncoder, 0, 100);
      cmd = {CMD_VELOCIDADE, velocidadeEncoder};
      xQueueSend(filaComandos, &cmd, 0); // sem bloquear: descarta se fila cheia
    }
    ultimoA = curA;

    // --- Botão encoder → toggle esteira ---
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

    // --- Botão extra → dispara eixo X ---
    int lBtnX = digitalRead(pinoBotaoExtra);
    if (lBtnX != ultimoBtnX) tBtnX = millis();
    if ((millis() - tBtnX) > 50 && lBtnX == LOW) {
      cmd = {CMD_DISPARA_X, 0};
      xQueueSend(filaComandos, &cmd, 0);
      // aguarda soltar o botão antes de continuar
      while (digitalRead(pinoBotaoExtra) == LOW) {
        vTaskDelay(pdMS_TO_TICKS(10));
      }
    }
    ultimoBtnX = lBtnX;

    // Libera CPU por 50 ms (padrão do lab)
    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// ============================================================
//  TASK 2: TaskControle  (Prioridade 2 — média)
//
//  Consumidora de comandos. SEMPRE bloqueia na fila com
//  portMAX_DELAY — nunca consome CPU em espera ativa.
//  Atualiza as variáveis volatile que a TaskMotorZ lê.
// ============================================================
void TaskControle(void *pvParameters) {
  (void)pvParameters;

  Comando cmdRecebido;
  bool ligado     = false;
  int  velocidade = 50;

  // Estado seguro inicial
  motorZLigado = false;
  motorZAtraso = 650;
  digitalWrite(enPinZ, HIGH);
  digitalWrite(enPinX, HIGH);

  atualizarLCD(ligado, velocidade);

  for (;;) {

    // Bloqueia aqui até chegar qualquer comando.
    // A TaskMotorZ (prio 3) cuida dos pulsos de forma independente.
    xQueueReceive(filaComandos, &cmdRecebido, portMAX_DELAY);

    switch (cmdRecebido.tipo) {

      case CMD_LIGA_Z:
        ligado       = true;
        motorZLigado = true;
        Serial.println(F("[CONTROLE] Esteira LIGADA"));
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_DESLIGA_Z:
        ligado       = false;
        motorZLigado = false;
        Serial.println(F("[CONTROLE] Esteira PARADA"));
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_TOGGLE_Z:
        ligado       = !ligado;
        motorZLigado = ligado;
        Serial.print(F("[CONTROLE] Toggle → "));
        Serial.println(ligado ? F("LIGADA") : F("PARADA"));
        atualizarLCD(ligado, velocidade);
        break;

      case CMD_DISPARA_X:
        // Pausa a esteira durante o movimento X
        motorZLigado = false;
        executarMovimentoX(velocidade);
        motorZLigado = ligado; // restaura estado anterior
        Serial.println(F("[CONTROLE] Eixo X concluido"));
        break;

      case CMD_VELOCIDADE:
        velocidade   = cmdRecebido.valor;
        motorZAtraso = map(velocidade, 0, 100, 1200, 100);
        Serial.print(F("[CONTROLE] Velocidade "));
        Serial.print(velocidade);
        Serial.print(F("% → "));
        Serial.print(motorZAtraso);
        Serial.println(F(" us"));
        atualizarLCD(ligado, velocidade);
        break;
    }
  }
}

// ============================================================
//  TASK 3: TaskMotorZ  (Prioridade 3 — alta)
//
//  Única responsabilidade: gerar pulsos STEP para o motor Z.
//  Lê as variáveis volatile definidas pela TaskControle.
//  Quando parado, cede CPU com vTaskDelay.
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
      // Cede CPU ao fim de cada pulso para tasks de mesma/maior prioridade
      // agirem. O tick do FreeRTOS no AVR é 15ms — sem este yield,
      // a task ficaria presa até o próximo tick mesmo com delayMicroseconds.
      taskYIELD();
    } else {
      digitalWrite(enPinZ, HIGH);
      vTaskDelay(pdMS_TO_TICKS(5));
    }
  }
}

// ============================================================
//  Sequência de movimento do Eixo X
//  Chamada de dentro da TaskControle — bloqueia até concluir.
// ============================================================
void executarMovimentoX(int velocidadePct) {
  // Quanto mais alta a velocidade, menor o atraso entre steps
  // Mínimo seguro: 100 µs. Fixo em 150 µs para movimento rápido e confiável.
  const int VEL_X = 150; // µs

  digitalWrite(enPinX, LOW);

  // Meia volta FRENTE
  digitalWrite(dirPinX, HIGH);
  for (int i = 0; i < PASSOS_MEIA_VOLTA; i++) {
    digitalWrite(stepPinX, HIGH);
    delayMicroseconds(2);       // pulso mínimo para o driver
    digitalWrite(stepPinX, LOW);
    delayMicroseconds(VEL_X);
  }

  vTaskDelay(pdMS_TO_TICKS(100)); // pausa mecânica antes de reverter

  // Meia volta TRÁS
  digitalWrite(dirPinX, LOW);
  for (int i = 0; i < PASSOS_MEIA_VOLTA; i++) {
    digitalWrite(stepPinX, HIGH);
    delayMicroseconds(2);
    digitalWrite(stepPinX, LOW);
    delayMicroseconds(VEL_X);
  }

  digitalWrite(enPinX, HIGH); // corta energia do motor X
}

// ============================================================
//  Atualiza o LCD  (chamada apenas da TaskControle)
//  Sem mutex necessário: só uma task acessa o LCD.
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
