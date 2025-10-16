// Copyright (c) 2025 Computer Vision Center (CVC) at the Universitat Autonoma
// de Barcelona (UAB).
//
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

#pragma once

#include "GameFramework/Character.h"

#include "WalkerBase.generated.h"

UCLASS()
class CARLA_API AWalkerBase : public ACharacter
{

  GENERATED_BODY()

  AWalkerBase(const FObjectInitializer &ObjectInitializer);
public:

  UPROPERTY(Category="Walker Base", BlueprintReadWrite, EditAnywhere)
  bool bAlive = true;

  UPROPERTY(Category="Walker Base", BlueprintReadWrite, EditAnywhere)
  float AfterLifeSpan = 10.0f;

  UPROPERTY(Category="Walker Base", BlueprintReadWrite, EditAnywhere)
  bool bUsesWheelChair = false;

  UFUNCTION(BlueprintCallable)
  void StartDeathLifeSpan()
  {
    SetLifeSpan(AfterLifeSpan);
  }

  // Called after adding wheelschairs as they have no tag.
  UFUNCTION(BlueprintCallable)
  void TagWheelchair();

};
